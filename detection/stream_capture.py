"""
Twitch stream frame capture using streamlink + ffmpeg.

Captures frames from a live Twitch stream at a configurable FPS
and feeds them to the death detector via a reader thread.

Uses streamlink to pipe the stream directly into ffmpeg, avoiding
the two-step URL resolution that can fail with certain Twitch configs.

Architecture:
  ffmpeg (target_fps) --[reader thread]--> _latest_frame (overwritten)
                                                |
  detect loop --[non-blocking grab]-------------+--> process
"""

import logging
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Quality fallback order — try requested quality first, then fall through
QUALITY_FALLBACK = [
    "best",
    "1080p60",
    "1080p",
    "720p60",
    "720p",
    "480p",
    "360p",
    "worst",
]


class StreamCapture:
    """Captures frames from a Twitch stream using streamlink piped to ffmpeg."""

    def __init__(
        self,
        channel: str,
        quality: str = "720p",
        target_fps: int = 15,
    ):
        self.channel = channel
        self.quality = quality
        self.target_fps = target_fps
        # streamlink works with both forms but www. is more reliable
        self.stream_url = f"https://www.twitch.tv/{channel}"

        self._streamlink_proc: subprocess.Popen | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self.last_error: str = ""

        # Reader thread state
        self._reader: threading.Thread | None = None
        self._frame_seq: int = 0        # monotonic counter, incremented on each new frame
        self._last_read_seq: int = 0    # seq of last frame returned by read_frame()
        self._new_frame_event = threading.Event()

    def _find_quality(self) -> str | None:
        """Find the best available quality from the stream."""
        # Build fallback list starting with requested quality
        candidates = [self.quality]
        for q in QUALITY_FALLBACK:
            if q not in candidates:
                candidates.append(q)

        # Use streamlink to check available streams
        try:
            result = subprocess.run(
                ["streamlink", self.stream_url],
                capture_output=True,
                text=True,
                timeout=20,
            )
            output = result.stdout + result.stderr
            logger.info("streamlink available streams output:\n%s", output.strip())

            # Pick the first candidate that appears in the output
            for quality in candidates:
                if quality in output:
                    if quality != self.quality:
                        logger.info(
                            "Requested %s not available, using %s",
                            self.quality,
                            quality,
                        )
                    return quality

            # If none matched explicitly, just try "best"
            return "best"

        except FileNotFoundError:
            self.last_error = "streamlink not found — install it with: pip install streamlink"
            logger.error(self.last_error)
            return None
        except subprocess.TimeoutExpired:
            self.last_error = "Timed out checking stream availability"
            logger.error(self.last_error)
            return None

    def start(self) -> bool:
        """
        Start capturing frames from the stream.

        Uses streamlink in pipe/stdout mode to feed raw video into ffmpeg,
        which is more reliable than resolving a URL first.
        """
        quality = self._find_quality()
        if not quality:
            return False

        # Reset sequence counters
        self._frame_seq = 0
        self._last_read_seq = 0
        self._new_frame_event.clear()

        # streamlink pipes the stream to stdout
        try:
            self._streamlink_proc = subprocess.Popen(
                [
                    "streamlink",
                    "--stdout",
                    "--loglevel", "warning",
                    self.stream_url,
                    quality,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.last_error = "streamlink not found — install it with: pip install streamlink"
            logger.error(self.last_error)
            return False

        # Give streamlink a moment to connect or fail
        time.sleep(3)
        if self._streamlink_proc.poll() is not None:
            stderr = ""
            if self._streamlink_proc.stderr:
                stderr = self._streamlink_proc.stderr.read().decode(errors="replace").strip()
            self.last_error = f"streamlink failed to connect: {stderr or 'channel may be offline'}"
            logger.error(self.last_error)
            self._streamlink_proc = None
            return False

        # ffmpeg reads from streamlink's stdout pipe
        self._ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-i", "pipe:0",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-vf", f"fps={self.target_fps},scale=1280:720",
                "-an",
                "-sn",
                "-loglevel", "error",
                "pipe:1",
            ],
            stdin=self._streamlink_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give ffmpeg a moment to start decoding
        time.sleep(2)
        if self._ffmpeg_proc.poll() is not None:
            stderr = ""
            if self._ffmpeg_proc.stderr:
                stderr = self._ffmpeg_proc.stderr.read().decode(errors="replace").strip()
            self.last_error = f"ffmpeg failed to decode stream: {stderr}"
            logger.error(self.last_error)
            self._cleanup()
            return False

        self._running = True
        self.last_error = ""

        # Start the reader thread that continuously ingests frames
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        logger.info(
            "Stream capture started for %s (%s quality, %d fps)",
            self.channel,
            quality,
            self.target_fps,
        )
        return True

    def _reader_loop(self) -> None:
        """Continuously read frames from ffmpeg stdout into _latest_frame."""
        width, height = 1280, 720
        frame_size = width * height * 3  # BGR24

        while self._running:
            if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
                break

            try:
                raw = self._ffmpeg_proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    # Stream ended or pipe broken
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
                with self._lock:
                    self._latest_frame = frame.copy()
                    self._frame_seq += 1
                    self._new_frame_event.set()

            except Exception as e:
                logger.error("Reader thread error: %s", e)
                break

        logger.debug("Reader thread exiting")

    def read_frame(self) -> np.ndarray | None:
        """
        Non-blocking read: returns the latest unread frame, or None if no
        new frame is available since the last call.
        """
        with self._lock:
            if self._frame_seq == self._last_read_seq:
                return None
            self._last_read_seq = self._frame_seq
            self._new_frame_event.clear()
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def wait_for_frame(self, timeout: float = 0.1) -> np.ndarray | None:
        """
        Block until a new frame arrives (or timeout expires), then return it.
        Uses threading.Event for efficient waiting instead of busy-polling.
        """
        self._new_frame_event.wait(timeout=timeout)
        return self.read_frame()

    def get_latest_frame(self) -> np.ndarray | None:
        """Return the most recently captured frame (thread-safe)."""
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def is_running(self) -> bool:
        """Check if the capture pipeline is still alive."""
        if self._ffmpeg_proc is None:
            return False
        return self._ffmpeg_proc.poll() is None

    def _cleanup(self) -> None:
        """Terminate both processes."""
        for proc in (self._ffmpeg_proc, self._streamlink_proc):
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._ffmpeg_proc = None
        self._streamlink_proc = None

    def stop(self) -> None:
        """Stop the stream capture."""
        self._running = False
        self._cleanup()
        # Wake up any thread waiting on new_frame_event so it can exit
        self._new_frame_event.set()
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        logger.info("Stream capture stopped for %s", self.channel)
