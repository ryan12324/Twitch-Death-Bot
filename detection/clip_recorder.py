"""
Death clip recorder.

Maintains a rolling buffer of recent frames so that when a death is
detected, it can save the seconds leading up to (and after) the death
as a video file. Perfect for building death compilations.
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLIPS_DIR = Path(__file__).parent.parent / "clips"


class ClipRecorder:
    def __init__(
        self,
        enabled: bool = True,
        pre_death_seconds: float = 3.0,
        post_death_seconds: float = 2.0,
        capture_fps: float = 0.5,
        output_dir: Path | None = None,
        output_fps: float = 10.0,
    ):
        """
        Args:
            enabled: Whether clip recording is active.
            pre_death_seconds: Seconds of footage to keep before the death.
            post_death_seconds: Seconds of footage to capture after the death.
            capture_fps: How many frames per second are being fed in
                         (should match 1/CAPTURE_INTERVAL).
            output_dir: Where to save clip files.
            output_fps: Playback FPS for saved clips. Frames are duplicated
                        to fill the timeline so clips play at readable speed.
        """
        self.enabled = enabled
        self.pre_death_seconds = pre_death_seconds
        self.post_death_seconds = post_death_seconds
        self.capture_fps = capture_fps
        self.output_fps = output_fps
        self.output_dir = output_dir or CLIPS_DIR

        # Rolling buffer: stores (timestamp, frame) tuples
        buffer_size = max(1, int(pre_death_seconds * capture_fps))
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # Post-death capture state
        self._capturing_post: bool = False
        self._post_frames: list[tuple[float, np.ndarray]] = []
        self._post_capture_deadline: float = 0.0
        self._death_number: int = 0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Clip recorder enabled: %.1fs pre-death, %.1fs post-death, "
                "saving to %s",
                pre_death_seconds,
                post_death_seconds,
                self.output_dir,
            )

    def feed_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the rolling buffer."""
        if not self.enabled:
            return

        now = time.time()

        with self._lock:
            self._buffer.append((now, frame.copy()))

            # If we're capturing post-death frames, collect them
            if self._capturing_post:
                self._post_frames.append((now, frame.copy()))
                if now >= self._post_capture_deadline:
                    self._capturing_post = False
                    self._save_clip()

    def on_death(self, death_number: int) -> None:
        """
        Called when a death is detected. Begins post-death capture.
        The pre-death frames are already in the rolling buffer.
        """
        if not self.enabled:
            return

        with self._lock:
            self._death_number = death_number
            self._post_frames = []
            self._capturing_post = True
            self._post_capture_deadline = (
                time.time() + self.post_death_seconds
            )

        logger.info(
            "Death #%d — recording %.1fs post-death clip...",
            death_number,
            self.post_death_seconds,
        )

    def flush(self) -> None:
        """Force-save any in-progress clip (e.g. on shutdown)."""
        with self._lock:
            if self._capturing_post and self._post_frames:
                self._capturing_post = False
                self._save_clip()

    def _save_clip(self) -> None:
        """Write the buffered frames to a video file. Must hold self._lock."""
        # Combine pre-death buffer + post-death frames
        all_frames = list(self._buffer) + self._post_frames

        if not all_frames:
            logger.warning("No frames to save for clip")
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"death_{self._death_number:04d}_{timestamp}.mp4"
        filepath = self.output_dir / filename

        # Get frame dimensions from first frame
        _, sample = all_frames[0]
        h, w = sample.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, self.output_fps, (w, h))

        if not writer.isOpened():
            logger.error("Failed to open video writer for %s", filepath)
            return

        # Calculate how many times to duplicate each frame so the clip
        # plays at output_fps speed over the real time duration
        if len(all_frames) >= 2:
            real_duration = all_frames[-1][0] - all_frames[0][0]
        else:
            real_duration = 1.0

        total_output_frames = max(1, int(real_duration * self.output_fps))
        duplication = max(1, total_output_frames // len(all_frames))

        # Write frames with timestamp overlay
        start_time = all_frames[0][0]
        for ts, frame in all_frames:
            # Add timestamp overlay
            elapsed = ts - start_time
            total_pre = self.pre_death_seconds
            relative = elapsed - total_pre
            label = f"{relative:+.1f}s"

            display = frame.copy()
            cv2.putText(
                display,
                label,
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            for _ in range(duplication):
                writer.write(display)

        writer.release()

        clip_length = real_duration
        logger.info(
            "Saved death clip: %s (%.1fs, %d source frames)",
            filepath,
            clip_length,
            len(all_frames),
        )

    def get_clip_count(self) -> int:
        """Return how many clip files have been saved."""
        if not self.output_dir.exists():
            return 0
        return len(list(self.output_dir.glob("death_*.mp4")))
