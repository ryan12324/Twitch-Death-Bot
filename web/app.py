"""
Testing GUI web application.

Provides a browser-based dashboard to:
  - Point at a live Twitch stream or upload a video/image
  - Watch the detector analyze frames in real time
  - See per-strategy score breakdowns (template, color, brightness, fade, scene)
  - Tune threshold / cooldown / game profile on the fly
  - View death log and session stats
"""

import base64
import io
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.counter import DeathCounter
from detection.detector import DeathDetector
from detection.stream_capture import StreamCapture
from game_profiles.profiles import (
    PROFILES, get_profile, list_profiles,
    profile_to_dict, save_profile, delete_profile, _parse_profile,
)

PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "game_profiles" / "templates"

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("web_gui")

app = Flask(__name__, template_folder="templates", static_folder="static")

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
        "color": 0.0,
        "brightness": 0.0,
        "fade": 0.0,
        "scene_change": 0.0,
        "confidence": 0.0,
    }


# ---------------------------------------------------------------------------
# Background detection loop
# ---------------------------------------------------------------------------


def _detection_thread():
    """Background thread that reads frames and runs detection."""
    logger.info("Detection thread started")
    from collections import deque
    fps_timestamps: deque[float] = deque(maxlen=60)

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

        frame = capture.read_frame()
        if frame is None:
            frame = capture.wait_for_frame(0.1)
            if frame is None:
                continue

        is_death, confidence = detector.analyze_frame(frame)
        scores = _get_scores(detector)

        # Draw detection overlay on the frame
        display = frame.copy()
        h, w = display.shape[:2]
        profile = detector.profile
        configured = scores.get("configured", [])

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
                # Add green tint to matching pixels without full-frame alloc
                pixels = display[idx].astype(np.int16)
                pixels[:, 1] = np.clip(pixels[:, 1] + 40, 0, 255)
                display[idx] = pixels.astype(np.uint8)

        # ── Draw template match box (if detector tracked it) ──
        if hasattr(detector, '_last_match_loc') and detector._last_match_loc:
            loc, size, score_val = detector._last_match_loc
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

        # ── Draw text detection boxes (if detector tracked them) ──
        if hasattr(detector, '_last_text_boxes') and detector._last_text_boxes:
            for bbox, text, conf_val, matched in detector._last_text_boxes:
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

        # Draw bar fills and labels directly (no per-bar frame copies)
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

        # ── FPS counter ──
        fps_timestamps.append(time.time())
        if len(fps_timestamps) >= 2:
            elapsed = fps_timestamps[-1] - fps_timestamps[0]
            fps = (len(fps_timestamps) - 1) / elapsed if elapsed > 0 else 0
        else:
            fps = 0
        y_offset += 28
        cv2.putText(
            display, f"{fps:.1f} fps",
            (bar_x, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
        )

        # ── Death flash ──
        if is_death:
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(
                display, "DEATH DETECTED",
                (w // 2 - 180, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
            )

        jpg = _encode_frame_jpg(display)

        with state_lock:
            state["latest_frame_jpg"] = jpg
            state["latest_scores"] = scores
            state["latest_confidence"] = conf
            state["frame_count"] += 1

            if is_death:
                session_count = counter.record_death(game, confidence)
                state["death_log"].append({
                    "time": time.strftime("%H:%M:%S"),
                    "session_count": session_count,
                    "total_count": counter.total_deaths,
                    "confidence": round(confidence, 3),
                })

    logger.info("Detection thread stopped")


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

    stop_event.clear()

    t = threading.Thread(target=_detection_thread, daemon=True)
    t.start()

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


@app.route("/video_feed")
def video_feed():
    """MJPEG stream for live preview."""
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
    time.sleep(0.5)
    stop_event.clear()
    return summary


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
