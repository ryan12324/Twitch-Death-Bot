"""
Twitch stream frame capture using streamlink + OpenCV.

Captures frames from a live Twitch stream at a configurable interval
and feeds them to the death detector.
"""

import logging
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StreamCapture:
    """Captures frames from a Twitch stream using streamlink."""

    def __init__(
        self,
        channel: str,
        quality: str = "720p",
        capture_interval: float = 2.0,
    ):
        self.channel = channel
        self.quality = quality
        self.capture_interval = capture_interval
        self.stream_url = f"https://twitch.tv/{channel}"

        self._process: subprocess.Popen | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

    def _get_stream_url(self) -> str | None:
        """Use streamlink to resolve the actual stream URL."""
        try:
            result = subprocess.run(
                [
                    "streamlink",
                    "--stream-url",
                    self.stream_url,
                    self.quality,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                logger.info("Resolved stream URL for %s", self.channel)
                return url
            else:
                logger.error(
                    "streamlink failed: %s", result.stderr.strip()
                )
                return None
        except FileNotFoundError:
            logger.error(
                "streamlink not found. Install it: pip install streamlink"
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error("Timed out resolving stream URL")
            return None

    def start(self) -> bool:
        """Start capturing frames from the stream. Returns True if started."""
        resolved_url = self._get_stream_url()
        if not resolved_url:
            logger.error(
                "Could not resolve stream for %s. Is the channel live?",
                self.channel,
            )
            return False

        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-i", resolved_url,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-vf", "fps=1/{},scale=1280:720".format(
                    max(1, int(self.capture_interval))
                ),
                "-an",        # no audio
                "-sn",        # no subtitles
                "-loglevel", "error",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._running = True
        logger.info(
            "Stream capture started for %s at %s quality",
            self.channel,
            self.quality,
        )
        return True

    def read_frame(self) -> np.ndarray | None:
        """Read a single frame from the stream. Returns None if unavailable."""
        if self._process is None or self._process.stdout is None:
            return None

        width, height = 1280, 720
        frame_size = width * height * 3  # BGR24

        try:
            raw = self._process.stdout.read(frame_size)
            if len(raw) != frame_size:
                return None

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (height, width, 3)
            )
            with self._lock:
                self._latest_frame = frame.copy()
            return frame

        except Exception as e:
            logger.error("Error reading frame: %s", e)
            return None

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recently captured frame (thread-safe)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def is_running(self) -> bool:
        """Check if the capture process is still alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def stop(self) -> None:
        """Stop the stream capture."""
        self._running = False
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            logger.info("Stream capture stopped for %s", self.channel)
