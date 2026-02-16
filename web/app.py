"""
Testing GUI web application.

Provides a browser-based dashboard to:
  - Point at a live Twitch stream or upload a video/image
  - Watch the detector analyze frames in real time
  - See per-strategy score breakdowns (template, color, brightness, fade, scene)
  - Tune threshold / cooldown / game profile on the fly
  - View death log and session stats

Video is streamed as JPEG frames via WebSocket for low-latency ~30fps playback,
while detection runs asynchronously at whatever rate the detector can manage.
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask_sock import Sock
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.twitch_bot import DeathBot
from detection.clip_recorder import ClipRecorder
from detection.counter import DeathCounter
from detection.death_db import DeathDatabase
from detection.detector import DeathDetector
from detection.stream_capture import StreamCapture
from bot.twitch_api import get_channel_game, match_profile_by_game_id
from game_profiles.profiles import (
    PROFILES, get_profile, list_profiles,
    profile_to_dict, save_profile, delete_profile, _parse_profile,
)
import queue

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "game_profiles" / "templates"

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("web_gui")

app = Flask(__name__, template_folder="templates", static_folder="static")
sock = Sock(app)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

state = {
    "detector": None,
    "capture": None,
    "counter": DeathCounter(),
    "game": "generic",
    "profile_name": "generic",
    "threshold": 0.80,
    "cooldown": 15,
    "running": False,
    "latest_frame_jpg": None,
    "latest_scores": {},
    "latest_confidence": 0.0,
    "death_log": [],        # [{time, session_count, total_count, confidence}]
    "frame_count": 0,
    "ws_clients": [],       # list[queue.Queue] — JPEG frames pushed to WS viewers
    "ws_clients_lock": threading.Lock(),
    "overlay_state": {
        "scores": {},
        "confidence": 0.0,
        "is_death": False,
        "death_flash_until": 0.0,
        "match_loc": None,
        "text_boxes": [],
        "configured": [],
    },
    # Bot control panel state
    "clip_recorder": None,      # ClipRecorder instance
    "bot": None,                # DeathBot instance
    "bot_loop": None,           # asyncio event loop for bot thread
    "bot_thread": None,         # threading.Thread running bot
    "messages_enabled": True,   # toggle for death announcements
    "channel": "",              # active channel name
    # SQLite death database (used by bot control panel only)
    "death_db": DeathDatabase(),
    "db_session_id": None,
}
state_lock = threading.Lock()
stop_event = threading.Event()


def _encode_frame_jpg(frame: np.ndarray, quality: int = 70) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _get_scores(detector: DeathDetector) -> dict:
    """Return the individual scores from the detector's last analyze_frame call."""
    return detector.last_scores or {
        "template": 0.0,
        "text": 0.0,
        "embedding": 0.0,
        "color": 0.0,
        "brightness": 0.0,
        "fade": 0.0,
        "scene_change": 0.0,
        "confidence": 0.0,
    }


# ---------------------------------------------------------------------------
# Drawing helpers — used by the video thread to render overlays
# ---------------------------------------------------------------------------


def _draw_overlays(frame: np.ndarray, overlay: dict, detector: DeathDetector) -> np.ndarray:
    """Draw detection overlays on a frame using cached overlay_state."""
    display = frame.copy()
    h, w = display.shape[:2]
    profile = detector.profile
    scores = overlay.get("scores", {})
    configured = overlay.get("configured", [])

    # ── Draw screen regions ──
    if profile.screen_regions:
        for region in profile.screen_regions:
            x1 = int(region.x_min * w)
            y1 = int(region.y_min * h)
            x2 = int(region.x_max * w)
            y2 = int(region.y_max * h)
            cv2.rectangle(display, (x1, y1), (x2, y2), (187, 102, 255), 2)

    # ── Draw color mask highlights (semi-transparent green) ──
    if profile.dominant_colors and "color" in configured:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for cr in profile.dominant_colors:
            lower = np.array(cr.lower, dtype=np.uint8)
            upper = np.array(cr.upper, dtype=np.uint8)
            combined_mask |= cv2.inRange(frame, lower, upper)
        idx = combined_mask > 0
        if np.any(idx):
            pixels = display[idx].astype(np.int16)
            pixels[:, 1] = np.clip(pixels[:, 1] + 40, 0, 255)
            display[idx] = pixels.astype(np.uint8)

    # ── Draw template match box ──
    match_loc = overlay.get("match_loc")
    if match_loc:
        loc, size, score_val = match_loc
        match_color = (0, 255, 0) if score_val > 0.75 else (0, 165, 255)
        cv2.rectangle(
            display,
            (loc[0], loc[1]),
            (loc[0] + size[0], loc[1] + size[1]),
            match_color, 2,
        )
        cv2.putText(
            display,
            f"tmpl: {score_val:.3f}",
            (loc[0], loc[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, match_color, 1,
        )

    # ── Draw text detection boxes ──
    text_boxes = overlay.get("text_boxes", [])
    for bbox, text, conf_val, matched in text_boxes:
        color = (0, 255, 0) if matched else (128, 128, 128)
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(display, [pts], True, color, 2)
        cv2.putText(
            display,
            f"{text} ({conf_val:.2f})",
            (pts[0][0], pts[0][1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )

    # ── Score bars on the right side ──
    bar_x = w - 200
    bar_w = 180
    bar_h = 18
    y_start = 20
    bar_items = [
        ("template", (180, 120, 0)),
        ("text", (0, 140, 255)),
        ("embedding", (205, 188, 0)),
        ("color", (0, 180, 0)),
        ("brightness", (0, 180, 180)),
        ("fade", (180, 0, 180)),
        ("scene_change", (0, 100, 255)),
    ]
    conf = scores.get("confidence", 0)
    threshold = detector.threshold

    # Single ROI-based alpha blend for the entire panel background
    panel_h = len(bar_items) * (bar_h + 4) + 8 + 24 + 22
    px1 = max(0, bar_x - 4)
    py1 = max(0, y_start - 4)
    px2 = min(w, bar_x + bar_w + 4)
    py2 = min(h, y_start + panel_h + 4)
    roi = display[py1:py2, px1:px2]
    dark = np.full_like(roi, (30, 30, 30), dtype=np.uint8)
    cv2.addWeighted(roi, 0.3, dark, 0.7, 0, roi)

    y_offset = y_start
    for label, color in bar_items:
        score_val = scores.get(label, 0)
        is_conf = label in configured
        if is_conf:
            fill_w = int(bar_w * min(1.0, score_val))
            cv2.rectangle(display, (bar_x, y_offset), (bar_x + fill_w, y_offset + bar_h), color, -1)
        label_color = (255, 255, 255) if is_conf else (100, 100, 100)
        txt = f"{label}: {score_val:.2f}" if is_conf else f"{label}: n/a"
        cv2.putText(display, txt, (bar_x + 4, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)
        y_offset += bar_h + 4

    # ── Confidence bar ──
    y_offset += 4
    conf_color = (0, 0, 255) if conf >= threshold else (100, 100, 100)
    cv2.rectangle(
        display, (bar_x, y_offset),
        (bar_x + int(bar_w * min(1.0, conf)), y_offset + 24),
        conf_color, -1,
    )
    tx = bar_x + int(bar_w * threshold)
    cv2.line(display, (tx, y_offset), (tx, y_offset + 24), (0, 255, 255), 2)
    cv2.putText(
        display, f"CONF: {conf:.3f} (thr: {threshold})",
        (bar_x + 4, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
    )

    # ── Death flash ──
    if overlay.get("death_flash_until", 0) > time.time():
        cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 8)
        cv2.putText(
            display, "DEATH DETECTED",
            (w // 2 - 180, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
        )

    return display


# ---------------------------------------------------------------------------
# Background detection loop — runs at detector speed (2-10fps)
# ---------------------------------------------------------------------------


def _detection_thread():
    """Background thread that reads frames and runs detection (no overlay drawing)."""
    logger.info("Detection thread started")

    while not stop_event.is_set():
        with state_lock:
            capture = state["capture"]
            detector = state["detector"]
            counter = state["counter"]
            game = state["game"]

        if capture is None or detector is None:
            time.sleep(0.5)
            continue

        if not capture.is_running():
            time.sleep(1)
            continue

        frame = capture.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        is_death, confidence = detector.analyze_frame(frame)
        scores = _get_scores(detector)

        # Build overlay state from detector results
        match_loc = None
        if hasattr(detector, '_last_match_loc') and detector._last_match_loc:
            match_loc = detector._last_match_loc

        text_boxes = []
        if hasattr(detector, '_last_text_boxes') and detector._last_text_boxes:
            text_boxes = list(detector._last_text_boxes)

        configured = scores.get("configured", [])

        with state_lock:
            state["latest_scores"] = scores
            state["latest_confidence"] = confidence
            state["frame_count"] += 1

            state["overlay_state"] = {
                "scores": scores,
                "confidence": confidence,
                "is_death": is_death,
                "death_flash_until": (time.time() + 1.0) if is_death else state["overlay_state"].get("death_flash_until", 0.0),
                "match_loc": match_loc,
                "text_boxes": text_boxes,
                "configured": configured,
            }

            if is_death:
                session_count = counter.record_death(game, confidence)
                state["death_log"].append({
                    "time": time.strftime("%H:%M:%S"),
                    "session_count": session_count,
                    "total_count": counter.total_deaths,
                    "confidence": round(confidence, 3),
                })

                # Clip recording
                clip_recorder = state.get("clip_recorder")
                clip_filename = None
                if clip_recorder:
                    clip_filename = f"death_{session_count:04d}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    clip_recorder.on_death(session_count)

                # SQLite death record
                db = state.get("death_db")
                if db:
                    db.record_death(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        session_count=session_count,
                        total_count=counter.total_deaths,
                        confidence=confidence,
                        game=game,
                        channel=state.get("channel", ""),
                        clip_filename=clip_filename,
                    )

                # Bot death announcement
                if state.get("messages_enabled", False):
                    bot = state.get("bot")
                    bot_loop = state.get("bot_loop")
                    if bot and bot_loop and bot_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            bot.announce_death(session_count, counter.total_deaths),
                            bot_loop,
                        )

    logger.info("Detection thread stopped")


# ---------------------------------------------------------------------------
# Video encoding loop — runs at ~30fps
# ---------------------------------------------------------------------------


def _video_thread():
    """Background thread that reads frames at ~30fps, draws overlays, sends JPEG over WS."""
    logger.info("Video thread started")
    from collections import deque
    fps_timestamps: deque[float] = deque(maxlen=60)
    frame_interval = 1.0 / 30.0
    target_w, target_h = 1280, 720

    while not stop_event.is_set():
        loop_start = time.time()

        with state_lock:
            capture = state["capture"]
            detector = state["detector"]
            overlay = dict(state["overlay_state"])
            clients_lock = state["ws_clients_lock"]

        if capture is None or detector is None:
            time.sleep(0.1)
            continue

        if not capture.is_running():
            time.sleep(0.1)
            continue

        frame = capture.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        # Resize to target dimensions if needed
        fh, fw = frame.shape[:2]
        if fw != target_w or fh != target_h:
            frame = cv2.resize(frame, (target_w, target_h))

        # Feed raw frame to clip recorder (before overlays)
        with state_lock:
            clip_rec = state.get("clip_recorder")
        if clip_rec:
            clip_rec.feed_frame(frame)

        # Draw overlays using cached detection results
        display = _draw_overlays(frame, overlay, detector)

        # FPS counter overlay
        fps_timestamps.append(time.time())
        if len(fps_timestamps) >= 2:
            elapsed = fps_timestamps[-1] - fps_timestamps[0]
            fps = (len(fps_timestamps) - 1) / elapsed if elapsed > 0 else 0
        else:
            fps = 0
        bar_x = target_w - 200
        y_fps = 20 + 7 * (18 + 4) + 8 + 24 + 28 + 4
        cv2.putText(
            display, f"{fps:.1f} fps",
            (bar_x, y_fps + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        # Encode JPEG once, broadcast to all WS clients + store for MJPEG fallback
        jpg = _encode_frame_jpg(display)
        with state_lock:
            state["latest_frame_jpg"] = jpg

        with clients_lock:
            for q in state["ws_clients"]:
                try:
                    # Non-blocking put; drop if client is behind
                    if q.full():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            pass
                    q.put_nowait(jpg)
                except queue.Full:
                    pass

        # Pace to ~30fps
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("Video thread stopped")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    counter = DeathCounter()
    return render_template(
        "home.html",
        profiles=PROFILES,
        profile_count=len(PROFILES),
        total_deaths=counter.total_deaths,
        sessions=len(counter.data.get("sessions", [])),
    )


@app.route("/test")
def test_page():
    return render_template("test.html", profiles=list_profiles())


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start detection against a stream or uploaded video."""
    data = request.json or {}
    channel = data.get("channel", "").strip()
    profile_name = data.get("profile", "generic")
    threshold = float(data.get("threshold", 0.80))
    cooldown = int(data.get("cooldown", 15))
    quality = data.get("quality", "720p")
    target_fps = int(data.get("target_fps", 15))

    if not channel:
        return jsonify({"error": "Channel name is required"}), 400

    if profile_name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {profile_name}"}), 400

    # Stop any running capture first
    _stop_capture()

    profile = get_profile(profile_name)
    detector = DeathDetector(profile=profile, threshold=threshold, cooldown=cooldown)
    capture = StreamCapture(
        channel=channel, quality=quality, target_fps=target_fps
    )

    if not capture.start():
        error = capture.last_error or f"Could not connect to {channel}. Is the stream live?"
        return jsonify({"error": error}), 502

    counter = DeathCounter()
    counter.start_session(profile_name)

    with state_lock:
        state["detector"] = detector
        state["capture"] = capture
        state["counter"] = counter
        state["game"] = profile_name
        state["profile_name"] = profile_name
        state["threshold"] = threshold
        state["cooldown"] = cooldown
        state["running"] = True
        state["death_log"] = []
        state["frame_count"] = 0
        state["overlay_state"] = {
            "scores": {},
            "confidence": 0.0,
            "is_death": False,
            "death_flash_until": 0.0,
            "match_loc": None,
            "text_boxes": [],
            "configured": [],
        }

    stop_event.clear()

    # Start detection thread (runs at detector speed)
    td = threading.Thread(target=_detection_thread, daemon=True)
    td.start()

    # Start video thread (runs at ~30fps)
    tv = threading.Thread(target=_video_thread, daemon=True)
    tv.start()

    return jsonify({
        "status": "started",
        "channel": channel,
        "profile": profile_name,
        "templates_loaded": len(detector.templates),
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the current detection session."""
    summary = _stop_capture()
    return jsonify({"status": "stopped", "summary": summary})


@app.route("/api/status")
def api_status():
    """Return current detection status, scores, and death log."""
    with state_lock:
        templates_loaded = len(state["detector"].templates) if state["detector"] else 0
        return jsonify({
            "running": state["running"],
            "profile": state["profile_name"],
            "threshold": state["threshold"],
            "scores": state["latest_scores"],
            "confidence": state["latest_confidence"],
            "session_deaths": state["counter"].session_deaths,
            "total_deaths": state["counter"].total_deaths,
            "death_log": state["death_log"][-50:],
            "frame_count": state["frame_count"],
            "templates_loaded": templates_loaded,
        })


@app.route("/api/threshold", methods=["POST"])
def api_threshold():
    """Update detection threshold on the fly."""
    data = request.json or {}
    threshold = float(data.get("threshold", 0.80))
    with state_lock:
        if state["detector"]:
            state["detector"].threshold = threshold
            state["threshold"] = threshold
    return jsonify({"threshold": threshold})


@app.route("/api/analyze_image", methods=["POST"])
def api_analyze_image():
    """Analyze a single uploaded image for death detection."""
    if "image" not in request.files:
        logger.warning("analyze_image: no 'image' in request.files")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    logger.info(
        "analyze_image: received file=%s size=%d bytes",
        file.filename, len(img_bytes),
    )
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        logger.error("analyze_image: cv2.imdecode returned None")
        return jsonify({"error": "Could not decode image"}), 400

    profile_name = request.form.get("profile", "generic")
    threshold = float(request.form.get("threshold", 0.80))

    logger.info(
        "analyze_image: decoded %dx%d profile=%s threshold=%.2f",
        frame.shape[1], frame.shape[0], profile_name, threshold,
    )

    profile = get_profile(profile_name)
    detector = DeathDetector(
        profile=profile, threshold=threshold, cooldown=0, required_consecutive=1,
    )
    is_death, confidence, debug_images = detector.analyze_frame_debug(frame)
    scores = _get_scores(detector)

    # Encode debug images as base64 JPEGs
    debug_b64 = {}
    for name, img in debug_images.items():
        jpg_bytes = _encode_frame_jpg(img, quality=85)
        debug_b64[name] = base64.b64encode(jpg_bytes).decode("ascii")

    logger.info(
        "analyze_image: RESULT is_death=%s confidence=%.4f templates_loaded=%d scores=%s",
        is_death, confidence, len(detector.templates), scores,
    )

    return jsonify({
        "is_death": is_death,
        "confidence": round(confidence, 4),
        "scores": scores,
        "templates_loaded": len(detector.templates),
        "debug_images": debug_b64,
    })


@app.route("/api/analyze_clip", methods=["POST"])
def api_analyze_clip():
    """Analyze a Twitch clip for death detection across all frames."""
    data = request.get_json()
    if not data or not data.get("url"):
        return jsonify({"error": "No clip URL provided"}), 400

    url = data["url"].strip()
    profile_name = data.get("profile", "generic")
    threshold = float(data.get("threshold", 0.80))

    # Normalize URL — if it doesn't start with http, prepend https://www.twitch.tv/
    if not url.startswith("http"):
        url = "https://www.twitch.tv/" + url

    if profile_name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {profile_name}"}), 400

    logger.info("analyze_clip: url=%s profile=%s threshold=%.2f", url, profile_name, threshold)

    tmp_path = None
    try:
        # Download clip via yt-dlp to a temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            result = subprocess.run(
                ["yt-dlp", "--no-playlist", "--format", "best",
                 "--merge-output-format", "mp4", "-o", tmp_path, url],
                capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError:
            return jsonify({"error": "yt-dlp is not installed. Run: pip install yt-dlp"}), 400
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Clip download timed out (60s limit)"}), 400

        if result.returncode != 0:
            logger.error("yt-dlp failed: %s", result.stderr)
            return jsonify({"error": f"Failed to download clip: {result.stderr.strip()[-200:]}"}), 400

        # Extract frames using cv2.VideoCapture at ~2fps
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open downloaded clip"}), 400

        clip_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_duration = total_video_frames / clip_fps if clip_fps > 0 else 0

        # Sample at ~2fps
        frame_interval = max(1, int(clip_fps / 2))
        max_frames = 120

        frames = []
        frame_times = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                frame_times.append(frame_idx / clip_fps)
                if len(frames) >= max_frames:
                    break
            frame_idx += 1
        cap.release()

        if not frames:
            return jsonify({"error": "No frames could be extracted from clip"}), 400

        logger.info(
            "analyze_clip: extracted %d frames from %d total (%.1fs clip at %.1ffps)",
            len(frames), total_video_frames, clip_duration, clip_fps,
        )

        # Create detector and run analyze_frame on all frames to find the best one
        profile = get_profile(profile_name)
        detector = DeathDetector(
            profile=profile, threshold=threshold, cooldown=0, required_consecutive=1,
        )

        timeline = []
        best_confidence = -1.0
        best_frame_idx = 0

        for i, frame in enumerate(frames):
            is_death, confidence = detector.analyze_frame(frame)
            timeline.append({
                "time_sec": round(frame_times[i], 2),
                "confidence": round(confidence, 4),
            })
            # Reset death_frame_count since frames aren't consecutive
            detector.death_frame_count = 0
            if confidence > best_confidence:
                best_confidence = confidence
                best_frame_idx = i

        # Run analyze_frame_debug on the best-scoring frame only
        is_death, confidence, debug_images = detector.analyze_frame_debug(frames[best_frame_idx])
        scores = _get_scores(detector)

        # Encode debug images as base64 JPEGs
        debug_b64 = {}
        for name, img in debug_images.items():
            jpg_bytes = _encode_frame_jpg(img, quality=85)
            debug_b64[name] = base64.b64encode(jpg_bytes).decode("ascii")

        logger.info(
            "analyze_clip: RESULT is_death=%s confidence=%.4f best_frame_time=%.2fs",
            is_death, confidence, frame_times[best_frame_idx],
        )

        return jsonify({
            "is_death": is_death,
            "confidence": round(confidence, 4),
            "scores": scores,
            "templates_loaded": len(detector.templates),
            "debug_images": debug_b64,
            "timeline": timeline,
            "clip_duration": round(clip_duration, 2),
            "best_frame_time": round(frame_times[best_frame_idx], 2),
        })

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream for live preview (fallback)."""
    def generate():
        while True:
            with state_lock:
                jpg = state["latest_frame_jpg"]
            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket route for JPEG video streaming
# ---------------------------------------------------------------------------


@sock.route("/ws/video")
def ws_video(ws):
    """WebSocket endpoint: streams JPEG frames to client in real time."""
    client_queue: queue.Queue = queue.Queue(maxsize=3)

    # Register
    with state["ws_clients_lock"]:
        state["ws_clients"].append(client_queue)

    try:
        while True:
            try:
                jpg = client_queue.get(timeout=5.0)
            except Exception:
                with state_lock:
                    if not state["running"]:
                        break
                continue

            try:
                ws.send(jpg)
            except Exception:
                break
    finally:
        with state["ws_clients_lock"]:
            try:
                state["ws_clients"].remove(client_queue)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Profile CRUD routes
# ---------------------------------------------------------------------------


@app.route("/profiles")
def profiles_page():
    return render_template("profiles.html", profiles=list_profiles())


@app.route("/api/profiles")
def api_profiles_list():
    """List all profiles as JSON."""
    return jsonify({
        name: profile_to_dict(p) for name, p in PROFILES.items()
    })


@app.route("/api/profiles/<name>")
def api_profile_get(name):
    """Get a single profile as JSON."""
    if name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {name}"}), 404
    return jsonify(profile_to_dict(PROFILES[name]))


@app.route("/api/profiles", methods=["POST"])
def api_profile_save():
    """Create or update a profile."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    try:
        profile = save_profile(data)
        return jsonify(profile_to_dict(profile))
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/profiles/<name>", methods=["DELETE"])
def api_profile_delete(name):
    """Delete a profile."""
    try:
        delete_profile(name)
        return jsonify({"status": "deleted"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/profiles/<name>/templates")
def api_profile_templates(name):
    """List template image filenames for a profile."""
    if name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {name}"}), 404
    tdir = TEMPLATES_DIR / PROFILES[name].template_dir
    if not tdir.is_dir():
        return jsonify({"files": []})
    files = sorted(
        f.name for f in tdir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    return jsonify({"files": files})


@app.route("/api/profiles/<name>/templates", methods=["POST"])
def api_profile_template_upload(name):
    """Upload a template image for a profile."""
    if name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {name}"}), 404
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    profile = PROFILES[name]
    tdir = TEMPLATES_DIR / profile.template_dir
    tdir.mkdir(parents=True, exist_ok=True)

    file = request.files["image"]
    filename = Path(file.filename).name  # sanitize
    dest = tdir / filename
    file.save(str(dest))
    return jsonify({"status": "uploaded", "filename": filename})


@app.route("/api/profiles/<name>/templates/<filename>")
def api_profile_template_file(name, filename):
    """Serve a template image file."""
    if name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {name}"}), 404
    tdir = TEMPLATES_DIR / PROFILES[name].template_dir
    return send_from_directory(str(tdir), filename)


@app.route("/api/profiles/<name>/test", methods=["POST"])
def api_profile_test(name):
    """Test a profile against an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Allow testing with unsaved profile data
    profile_data_str = request.form.get("profile_data")
    if profile_data_str:
        try:
            data = json.loads(profile_data_str)
            profile = _parse_profile(data)
        except Exception as e:
            return jsonify({"error": f"Invalid profile data: {e}"}), 400
    else:
        if name not in PROFILES:
            return jsonify({"error": f"Unknown profile: {name}"}), 404
        profile = PROFILES[name]

    detector = DeathDetector(
        profile=profile, threshold=0.80, cooldown=0, required_consecutive=1,
    )
    is_death, confidence, debug_images = detector.analyze_frame_debug(frame)
    scores = _get_scores(detector)

    debug_b64 = {}
    for img_name, img in debug_images.items():
        jpg_bytes = _encode_frame_jpg(img, quality=85)
        debug_b64[img_name] = base64.b64encode(jpg_bytes).decode("ascii")

    return jsonify({
        "is_death": is_death,
        "confidence": round(confidence, 4),
        "scores": scores,
        "templates_loaded": len(detector.templates),
        "debug_images": debug_b64,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stop_capture() -> dict:
    stop_event.set()
    summary = {}
    with state_lock:
        if state["capture"]:
            state["capture"].stop()
            state["capture"] = None
        if state["counter"]:
            summary = state["counter"].end_session()
        state["detector"] = None
        state["running"] = False
        state["latest_frame_jpg"] = None
        # Clear bot state if any (test page doesn't start a bot, but be safe)
        state["clip_recorder"] = None
        state["channel"] = ""
    time.sleep(0.5)
    stop_event.clear()
    return summary


def _start_bot_thread(token: str, channel: str, counter: DeathCounter,
                      game: str, clip_recorder: ClipRecorder | None,
                      prefix: str = "!", announce_enabled: bool = True):
    """Create DeathBot and run it in a background thread with its own event loop."""
    loop = asyncio.new_event_loop()
    bot = DeathBot(
        token=token,
        prefix=prefix,
        channel=channel,
        counter=counter,
        game=game,
        clip_recorder=clip_recorder,
        announce_enabled=announce_enabled,
    )

    def _run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(bot.start())
        except Exception:
            logger.exception("Bot event loop error")
        finally:
            loop.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return bot, loop, thread


def _stop_bot_capture() -> dict:
    """Stop everything: capture, clip recorder, bot, clear all state."""
    stop_event.set()
    summary = {}

    with state_lock:
        # Stop stream capture
        if state["capture"]:
            state["capture"].stop()
            state["capture"] = None

        # Flush clip recorder
        clip_rec = state.get("clip_recorder")
        if clip_rec:
            clip_rec.flush()
            state["clip_recorder"] = None

        # End SQLite session
        db = state.get("death_db")
        db_session_id = state.get("db_session_id")
        if db and db_session_id:
            db.end_session(db_session_id, state["counter"].session_deaths)
            state["db_session_id"] = None

        # End counter session
        if state["counter"]:
            summary = state["counter"].end_session()

        state["detector"] = None
        state["running"] = False
        state["latest_frame_jpg"] = None
        state["channel"] = ""

        # Close DeathBot
        bot = state.get("bot")
        bot_loop = state.get("bot_loop")

    # Close bot outside state_lock to avoid deadlocks
    if bot and bot_loop and bot_loop.is_running():
        try:
            future = asyncio.run_coroutine_threadsafe(bot.close(), bot_loop)
            future.result(timeout=5)
        except Exception:
            logger.warning("Bot close timed out or errored")

    with state_lock:
        state["bot"] = None
        state["bot_loop"] = None
        state["bot_thread"] = None

    time.sleep(0.5)
    stop_event.clear()
    return summary


# ---------------------------------------------------------------------------
# Bot control panel routes
# ---------------------------------------------------------------------------

CLIPS_DIR = PROJECT_ROOT / "clips"


@app.route("/bot")
def bot_page():
    return render_template(
        "bot.html",
        profiles=list_profiles(),
        default_channel=os.getenv("TWITCH_CHANNEL", ""),
        default_token=os.getenv("TWITCH_TOKEN", ""),
    )


@app.route("/api/bot/start", methods=["POST"])
def api_bot_start():
    """Start stream + detection + clips + optional chat bot."""
    data = request.json or {}
    channel = data.get("channel", "").strip()
    profile_name = data.get("profile", "generic")
    threshold = float(data.get("threshold", 0.80))
    cooldown = int(data.get("cooldown", 15))
    quality = data.get("quality", "720p")
    target_fps = int(data.get("target_fps", 15))
    twitch_token = data.get("twitch_token", "").strip() or os.getenv("TWITCH_TOKEN", "")

    if not channel:
        return jsonify({"error": "Channel name is required"}), 400

    if profile_name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {profile_name}"}), 400

    # Stop any running session (test or bot)
    _stop_bot_capture()

    profile = get_profile(profile_name)
    detector = DeathDetector(profile=profile, threshold=threshold, cooldown=cooldown)
    capture = StreamCapture(channel=channel, quality=quality, target_fps=target_fps)

    if not capture.start():
        error = capture.last_error or f"Could not connect to {channel}. Is the stream live?"
        return jsonify({"error": error}), 502

    counter = DeathCounter()
    counter.start_session(profile_name)

    # Start SQLite session
    db_session_id = state["death_db"].start_session(profile_name, channel)

    # Initialize clip recorder
    clip_recorder = ClipRecorder(enabled=True, output_dir=CLIPS_DIR)

    with state_lock:
        state["detector"] = detector
        state["capture"] = capture
        state["counter"] = counter
        state["game"] = profile_name
        state["profile_name"] = profile_name
        state["threshold"] = threshold
        state["cooldown"] = cooldown
        state["running"] = True
        state["death_log"] = []
        state["frame_count"] = 0
        state["channel"] = channel
        state["clip_recorder"] = clip_recorder
        state["db_session_id"] = db_session_id
        state["messages_enabled"] = bool(twitch_token)
        state["overlay_state"] = {
            "scores": {},
            "confidence": 0.0,
            "is_death": False,
            "death_flash_until": 0.0,
            "match_loc": None,
            "text_boxes": [],
            "configured": [],
        }

    stop_event.clear()

    # Start detection thread
    td = threading.Thread(target=_detection_thread, daemon=True)
    td.start()

    # Start video thread
    tv = threading.Thread(target=_video_thread, daemon=True)
    tv.start()

    # Start bot if token provided
    bot_connected = False
    if twitch_token:
        bot, bot_loop, bot_thread = _start_bot_thread(
            twitch_token, channel, counter, profile.display_name, clip_recorder,
        )
        with state_lock:
            state["bot"] = bot
            state["bot_loop"] = bot_loop
            state["bot_thread"] = bot_thread
        bot_connected = True

    return jsonify({
        "status": "started",
        "channel": channel,
        "profile": profile_name,
        "templates_loaded": len(detector.templates),
        "bot_connected": bot_connected,
    })


@app.route("/api/bot/stop", methods=["POST"])
def api_bot_stop():
    """Stop the bot session (capture + clips + bot)."""
    summary = _stop_bot_capture()
    return jsonify({"status": "stopped", "summary": summary})


@app.route("/api/bot/status")
def api_bot_status():
    """Return bot status: same as /api/status + bot-specific fields."""
    with state_lock:
        templates_loaded = len(state["detector"].templates) if state["detector"] else 0
        clip_rec = state.get("clip_recorder")
        clip_count = clip_rec.get_clip_count() if clip_rec else 0
        bot = state.get("bot")
        bot_loop = state.get("bot_loop")
        bot_connected = bool(bot and bot_loop and bot_loop.is_running())
        return jsonify({
            "running": state["running"],
            "profile": state["profile_name"],
            "threshold": state["threshold"],
            "scores": state["latest_scores"],
            "confidence": state["latest_confidence"],
            "session_deaths": state["counter"].session_deaths,
            "total_deaths": state["counter"].total_deaths,
            "death_log": state["death_log"][-50:],
            "frame_count": state["frame_count"],
            "templates_loaded": templates_loaded,
            "messages_enabled": state.get("messages_enabled", False),
            "bot_connected": bot_connected,
            "clip_count": clip_count,
            "channel": state.get("channel", ""),
        })


@app.route("/api/bot/messages", methods=["POST"])
def api_bot_messages():
    """Toggle death announcements."""
    data = request.json or {}
    enabled = bool(data.get("enabled", True))
    with state_lock:
        state["messages_enabled"] = enabled
    return jsonify({"messages_enabled": enabled})


@app.route("/api/bot/clips")
def api_bot_clips():
    """List saved clips sorted newest first."""
    if not CLIPS_DIR.exists():
        return jsonify([])

    clips = []
    for f in sorted(CLIPS_DIR.glob("death_*.mp4"), reverse=True):
        # Parse death number from filename: death_0001_20260210_143022.mp4
        parts = f.stem.split("_")
        death_number = int(parts[1]) if len(parts) >= 2 else 0
        size_mb = round(f.stat().st_size / (1024 * 1024), 2)
        clips.append({
            "filename": f.name,
            "death_number": death_number,
            "timestamp": f.stat().st_mtime,
            "size_mb": size_mb,
            "url": f"/api/bot/clips/{f.name}",
        })

    return jsonify(clips)


@app.route("/api/bot/clips/<filename>")
def api_bot_clip_file(filename):
    """Serve a clip MP4 file."""
    if not CLIPS_DIR.exists():
        return jsonify({"error": "No clips directory"}), 404
    return send_from_directory(str(CLIPS_DIR), filename)


# ---------------------------------------------------------------------------
# Streamer config CRUD + game detection
# ---------------------------------------------------------------------------


@app.route("/api/streamers")
def api_streamers_list():
    """List all saved streamer configs."""
    db = state.get("death_db")
    if not db:
        return jsonify([])
    return jsonify(db.get_streamer_configs())


@app.route("/api/streamers", methods=["POST"])
def api_streamers_save():
    """Create or update a streamer config (upsert by channel)."""
    db = state.get("death_db")
    if not db:
        return jsonify({"error": "Database not available"}), 500
    data = request.get_json()
    if not data or not data.get("channel", "").strip():
        return jsonify({"error": "Channel name is required"}), 400
    try:
        saved = db.save_streamer_config(data)
        return jsonify(saved)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/streamers/<channel>", methods=["DELETE"])
def api_streamers_delete(channel):
    """Delete a streamer config."""
    db = state.get("death_db")
    if not db:
        return jsonify({"error": "Database not available"}), 500
    if db.delete_streamer_config(channel):
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Streamer not found"}), 404


@app.route("/api/twitch/game/<channel>")
def api_twitch_game(channel):
    """Detect what game a channel is playing via Twitch Helix API."""
    client_id = request.args.get("client_id", "") or os.getenv("TWITCH_CLIENT_ID", "")
    token = request.args.get("token", "") or os.getenv("TWITCH_TOKEN", "")

    if not client_id or not token:
        return jsonify({
            "error": "TWITCH_CLIENT_ID and TWITCH_TOKEN are required "
                     "(set as env vars or pass as query params)"
        }), 400

    result = get_channel_game(channel, client_id, token)
    if result is None:
        return jsonify({"game_id": None, "game_name": None, "suggested_profile": None})

    suggested = match_profile_by_game_id(result["game_id"])
    return jsonify({
        "game_id": result["game_id"],
        "game_name": result["game_name"],
        "suggested_profile": suggested,
    })


# ---------------------------------------------------------------------------
# Death Database API endpoints (SQLite-backed)
# ---------------------------------------------------------------------------


@app.route("/api/bot/db/deaths")
def api_db_deaths():
    """List deaths from SQLite DB, newest first."""
    db = state.get("death_db")
    if not db:
        return jsonify([])
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)
    game = request.args.get("game")
    return jsonify(db.get_deaths(limit=limit, offset=offset, game=game))


@app.route("/api/bot/db/deaths/<int:death_id>", methods=["DELETE"])
def api_db_death_delete(death_id):
    """Delete a death record."""
    db = state.get("death_db")
    if not db:
        return jsonify({"error": "Database not available"}), 500
    if db.delete_death(death_id):
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Death not found"}), 404


@app.route("/api/bot/db/deaths/<int:death_id>", methods=["PATCH"])
def api_db_death_update(death_id):
    """Update notes or clip_filename on a death record."""
    db = state.get("death_db")
    if not db:
        return jsonify({"error": "Database not available"}), 500
    data = request.json or {}
    kwargs = {}
    if "notes" in data:
        kwargs["notes"] = data["notes"]
    if "clip_filename" in data:
        kwargs["clip_filename"] = data["clip_filename"]
    if not kwargs:
        return jsonify({"error": "No updatable fields provided"}), 400
    if db.update_death(death_id, **kwargs):
        return jsonify(db.get_death(death_id))
    return jsonify({"error": "Death not found"}), 404


@app.route("/api/bot/db/stats")
def api_db_stats():
    """Aggregate death statistics."""
    db = state.get("death_db")
    if not db:
        return jsonify({"total_deaths": 0, "per_game": {}, "session_count": 0})
    game = request.args.get("game")
    return jsonify(db.get_stats(game=game))


@app.route("/api/bot/db/total_deaths", methods=["POST"])
def api_db_total_deaths():
    """Set total deaths override. Also syncs to the JSON counter."""
    db = state.get("death_db")
    if not db:
        return jsonify({"error": "Database not available"}), 500
    data = request.json or {}
    total = int(data.get("total_deaths", 0))
    db.set_total_deaths(total)
    # Sync to the JSON counter so bot chat commands reflect the override
    with state_lock:
        counter = state.get("counter")
        if counter and total > 0:
            counter.data["total_deaths"] = total
            counter._save()
    return jsonify({"total_deaths": db.get_total_deaths()})


@app.route("/api/bot/db/sessions")
def api_db_sessions():
    """List sessions from SQLite DB."""
    db = state.get("death_db")
    if not db:
        return jsonify([])
    limit = request.args.get("limit", 50, type=int)
    return jsonify(db.get_sessions(limit=limit))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    port = int(os.getenv("WEB_PORT", "4444"))
    debug = os.getenv("WEB_DEBUG", "false").lower() in ("true", "1")
    logger.info("Starting testing GUI on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
