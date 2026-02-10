"""
Audio capture from a Twitch stream using streamlink + ffmpeg.

Runs a separate lightweight pipeline that extracts only audio from the
stream as raw 16-bit signed little-endian mono PCM at 16 kHz.  Feeds
captured audio chunks to an AudioDetector for death-sound matching.

Architecture:
  streamlink (worst quality) --> ffmpeg (-vn, PCM s16le 16kHz mono) --> reader thread
                                                                           |
  AudioDetector.feed_audio() <---------------------------------------------+
"""

import logging
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# Audio format parameters — must match what AudioDetector expects
SAMPLE_RATE = 16000   # Hz
CHANNELS = 1          # mono
SAMPLE_WIDTH = 2      # bytes (16-bit)

# Read roughly this many seconds of audio per chunk
_CHUNK_SECONDS = 0.5
_CHUNK_BYTES = int(SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * _CHUNK_SECONDS)


class AudioCapture:
    """Captures audio from a Twitch stream and feeds it to an AudioDetector."""

    def __init__(self, channel: str, quality: str = "worst"):
        self.channel = channel
        self.quality = quality
        self.stream_url = f"https://www.twitch.tv/{channel}"

        self._streamlink_proc: subprocess.Popen | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._running = False
        self._reader: threading.Thread | None = None
        self._audio_callback = None
        self.last_error: str = ""

    def start(self, audio_callback) -> bool:
        """Start capturing audio and feeding chunks to *audio_callback*.

        The callback receives ``bytes`` of raw PCM data (s16le, mono, 16 kHz).
        Returns True on success.
        """
        self._audio_callback = audio_callback

        try:
            self._streamlink_proc = subprocess.Popen(
                [
                    "streamlink",
                    "--stdout",
                    "--loglevel", "warning",
                    self.stream_url,
                    self.quality,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            self.last_error = "streamlink not found"
            logger.error(self.last_error)
            return False

        # Give streamlink a moment to connect
        time.sleep(3)
        if self._streamlink_proc.poll() is not None:
            stderr = ""
            if self._streamlink_proc.stderr:
                stderr = self._streamlink_proc.stderr.read().decode(errors="replace").strip()
            self.last_error = f"streamlink audio failed: {stderr or 'channel may be offline'}"
            logger.error(self.last_error)
            self._streamlink_proc = None
            return False

        # FFmpeg extracts audio only as raw PCM
        self._ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-i", "pipe:0",
                "-vn",                          # discard video
                "-f", "s16le",                  # raw PCM output
                "-acodec", "pcm_s16le",
                "-ac", str(CHANNELS),
                "-ar", str(SAMPLE_RATE),
                "-loglevel", "error",
                "pipe:1",
            ],
            stdin=self._streamlink_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(2)
        if self._ffmpeg_proc.poll() is not None:
            stderr = ""
            if self._ffmpeg_proc.stderr:
                stderr = self._ffmpeg_proc.stderr.read().decode(errors="replace").strip()
            self.last_error = f"ffmpeg audio decode failed: {stderr}"
            logger.error(self.last_error)
            self._cleanup()
            return False

        self._running = True
        self.last_error = ""

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        logger.info(
            "Audio capture started for %s (%s quality, %d Hz mono)",
            self.channel, self.quality, SAMPLE_RATE,
        )
        return True

    def _reader_loop(self) -> None:
        """Continuously read PCM audio and forward to the callback."""
        while self._running:
            if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
                break
            try:
                data = self._ffmpeg_proc.stdout.read(_CHUNK_BYTES)
                if not data:
                    break
                if self._audio_callback:
                    self._audio_callback(data)
            except Exception as e:
                logger.error("Audio reader error: %s", e)
                break

        logger.debug("Audio reader thread exiting")

    def is_running(self) -> bool:
        if self._ffmpeg_proc is None:
            return False
        return self._ffmpeg_proc.poll() is None

    def _cleanup(self) -> None:
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
        self._running = False
        self._cleanup()
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        logger.info("Audio capture stopped for %s", self.channel)
