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
from flask import Flask, Response, jsonify, render_template, request
from dotenv import load_dotenv

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.counter import DeathCounter
from detection.detector import DeathDetector
from detection.stream_capture import StreamCapture
from game_profiles.profiles import PROFILES, get_profile, list_profiles

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


def _analyze_frame_detailed(detector: DeathDetector, frame: np.ndarray) -> dict:
    """Run all detector strategies and return individual scores."""
    template = detector._template_match_score(frame)
    color = detector._color_analysis_score(frame)
    brightness = detector._brightness_score(frame)
    fade = detector._fade_detection_score(frame)
    scene = detector._scene_change_score(frame)

    if detector.templates:
        confidence = (
            template * 0.40
            + color * 0.15
            + brightness * 0.15
            + fade * 0.15
            + scene * 0.15
        )
    else:
        confidence = (
            color * 0.30
            + brightness * 0.25
            + fade * 0.25
            + scene * 0.20
        )

    return {
        "template": round(template, 4),
        "color": round(color, 4),
        "brightness": round(brightness, 4),
        "fade": round(fade, 4),
        "scene_change": round(scene, 4),
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Background detection loop
# ---------------------------------------------------------------------------


def _detection_thread():
    """Background thread that reads frames and runs detection."""
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

        frame = capture.read_frame()
        if frame is None:
            time.sleep(0.2)
            continue

        scores = _analyze_frame_detailed(detector, frame)
        is_death, confidence = detector.analyze_frame(frame)

        # Draw detection overlay on the frame
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw score bars on the right side
        bar_x = w - 200
        bar_w = 180
        bar_h = 18
        y_offset = 20
        bar_colors = {
            "template": (180, 120, 0),
            "color": (0, 180, 0),
            "brightness": (0, 180, 180),
            "fade": (180, 0, 180),
            "scene_change": (0, 100, 255),
        }
        for label, score in scores.items():
            if label == "confidence":
                continue
            color = bar_colors.get(label, (200, 200, 200))
            # Background
            cv2.rectangle(
                display,
                (bar_x, y_offset),
                (bar_x + bar_w, y_offset + bar_h),
                (40, 40, 40),
                -1,
            )
            # Fill
            fill_w = int(bar_w * min(1.0, score))
            cv2.rectangle(
                display,
                (bar_x, y_offset),
                (bar_x + fill_w, y_offset + bar_h),
                color,
                -1,
            )
            # Label
            cv2.putText(
                display,
                f"{label}: {score:.2f}",
                (bar_x + 4, y_offset + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            y_offset += bar_h + 4

        # Confidence bar (wider)
        y_offset += 4
        conf = scores["confidence"]
        threshold = detector.threshold
        conf_color = (0, 0, 255) if conf >= threshold else (100, 100, 100)
        cv2.rectangle(
            display, (bar_x, y_offset), (bar_x + bar_w, y_offset + 24), (40, 40, 40), -1
        )
        cv2.rectangle(
            display,
            (bar_x, y_offset),
            (bar_x + int(bar_w * min(1.0, conf)), y_offset + 24),
            conf_color,
            -1,
        )
        # Threshold marker line
        tx = bar_x + int(bar_w * threshold)
        cv2.line(display, (tx, y_offset), (tx, y_offset + 24), (0, 255, 255), 2)
        cv2.putText(
            display,
            f"CONFIDENCE: {conf:.3f} (threshold: {threshold})",
            (bar_x + 4, y_offset + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        # Death flash
        if is_death:
            cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(
                display,
                "DEATH DETECTED",
                (w // 2 - 180, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
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
    capture_interval = float(data.get("capture_interval", 2.0))

    if not channel:
        return jsonify({"error": "Channel name is required"}), 400

    if profile_name not in PROFILES:
        return jsonify({"error": f"Unknown profile: {profile_name}"}), 400

    # Stop any running capture first
    _stop_capture()

    profile = get_profile(profile_name)
    detector = DeathDetector(profile=profile, threshold=threshold, cooldown=cooldown)
    capture = StreamCapture(
        channel=channel, quality=quality, capture_interval=capture_interval
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

    return jsonify({"status": "started", "channel": channel, "profile": profile_name})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the current detection session."""
    summary = _stop_capture()
    return jsonify({"status": "stopped", "summary": summary})


@app.route("/api/status")
def api_status():
    """Return current detection status, scores, and death log."""
    with state_lock:
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
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    profile_name = request.form.get("profile", "generic")
    threshold = float(request.form.get("threshold", 0.80))

    profile = get_profile(profile_name)
    detector = DeathDetector(profile=profile, threshold=threshold, cooldown=0)
    # Feed twice to satisfy consecutive frame requirement
    detector.analyze_frame(frame)
    scores = _analyze_frame_detailed(detector, frame)
    is_death, confidence = detector.analyze_frame(frame)

    return jsonify({
        "is_death": is_death,
        "confidence": round(confidence, 4),
        "scores": scores,
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
