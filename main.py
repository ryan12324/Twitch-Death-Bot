"""
Twitch Death Counter Bot - Main entry point.

Connects to a Twitch stream, captures frames, detects death screens
using image analysis, and announces deaths in chat.
"""

import asyncio
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from bot.twitch_bot import DeathBot
from detection.audio_capture import AudioCapture
from detection.audio_detector import AudioDetector
from detection.clip_recorder import ClipRecorder
from detection.counter import DeathCounter
from detection.detector import DeathDetector
from detection.stream_capture import StreamCapture
from game_profiles.profiles import get_profile, list_profiles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("death_bot")


def get_config() -> dict:
    """Read configuration from environment variables."""
    token = os.getenv("TWITCH_TOKEN", "")
    channel = os.getenv("TWITCH_CHANNEL", "")

    if not token or not channel:
        logger.error(
            "TWITCH_TOKEN and TWITCH_CHANNEL must be set. "
            "Copy config.example.env to .env and fill in your values."
        )
        sys.exit(1)

    game = os.getenv("GAME_PROFILE", "generic")
    if game not in list_profiles():
        logger.error(
            "Unknown GAME_PROFILE '%s'. Available: %s",
            game,
            ", ".join(list_profiles()),
        )
        sys.exit(1)

    # TARGET_FPS replaces the old CAPTURE_INTERVAL setting
    target_fps = int(os.getenv("TARGET_FPS", "15"))
    if os.getenv("CAPTURE_INTERVAL"):
        logger.warning(
            "CAPTURE_INTERVAL is deprecated — use TARGET_FPS instead. "
            "Ignoring CAPTURE_INTERVAL=%s, using TARGET_FPS=%d.",
            os.getenv("CAPTURE_INTERVAL"),
            target_fps,
        )

    return {
        "token": token,
        "channel": channel,
        "game": game,
        "target_fps": target_fps,
        "threshold": float(os.getenv("DETECTION_THRESHOLD", "0.80")),
        "cooldown": int(os.getenv("DEATH_COOLDOWN", "15")),
        "quality": os.getenv("STREAM_QUALITY", "720p"),
        "clip_enabled": os.getenv("CLIP_ENABLED", "true").lower() in ("true", "1", "yes"),
        "clip_pre_seconds": float(os.getenv("CLIP_PRE_DEATH_SECONDS", "3.0")),
        "clip_post_seconds": float(os.getenv("CLIP_POST_DEATH_SECONDS", "2.0")),
        "clip_output_fps": float(os.getenv("CLIP_OUTPUT_FPS", "10.0")),
        "clip_output_dir": os.getenv("CLIP_OUTPUT_DIR", ""),
    }


# ---------------------------------------------------------------------------
# Detection loop (runs in a background thread)
# ---------------------------------------------------------------------------


def detection_loop(
    capture: StreamCapture,
    detector: DeathDetector,
    counter: DeathCounter,
    clip_recorder: ClipRecorder,
    game: str,
    bot: DeathBot,
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """Continuously read frames and check for deaths."""
    logger.info("Detection loop started")

    while not stop_event.is_set():
        if not capture.is_running():
            logger.warning("Stream capture stopped. Attempting to reconnect...")
            if not capture.start():
                logger.error("Reconnect failed. Waiting 30s before retry...")
                stop_event.wait(30)
                continue

        frame = capture.read_frame()
        if frame is None:
            frame = capture.wait_for_frame(0.1)
            if frame is None:
                continue

        # Feed every frame to the clip recorder's rolling buffer
        clip_recorder.feed_frame(frame)

        is_death, confidence = detector.analyze_frame(frame)

        if is_death:
            session_count = counter.record_death(game, confidence)
            total_count = counter.total_deaths

            # Trigger clip save (captures post-death frames automatically)
            clip_recorder.on_death(session_count)

            # Schedule chat announcement on the bot's event loop
            asyncio.run_coroutine_threadsafe(
                bot.announce_death(session_count, total_count),
                loop,
            )

    # Flush any in-progress clip on shutdown
    clip_recorder.flush()
    logger.info("Detection loop stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = get_config()

    profile = get_profile(config["game"])
    logger.info("Game profile: %s", profile.display_name)

    # Initialize components
    counter = DeathCounter()
    counter.start_session(config["game"])

    # Set up audio detection if the profile has audio samples
    audio_detector: AudioDetector | None = None
    audio_capture: AudioCapture | None = None
    if profile.audio_samples_dir:
        audio_detector = AudioDetector(profile.audio_samples_dir)
        if audio_detector.reference_specs:
            audio_capture = AudioCapture(
                channel=config["channel"],
                quality="worst",
            )
            logger.info(
                "Audio detection enabled (%d reference sound(s))",
                len(audio_detector.reference_specs),
            )
        else:
            logger.warning(
                "Profile has audio_samples_dir='%s' but no WAV files found — "
                "audio detection disabled. Add .wav files to "
                "game_profiles/audio_samples/%s/",
                profile.audio_samples_dir,
                profile.audio_samples_dir,
            )
            audio_detector = None

    detector = DeathDetector(
        profile=profile,
        threshold=config["threshold"],
        cooldown=config["cooldown"],
        audio_detector=audio_detector,
    )

    capture = StreamCapture(
        channel=config["channel"],
        quality=config["quality"],
        target_fps=config["target_fps"],
    )

    clip_output_dir = Path(config["clip_output_dir"]) if config["clip_output_dir"] else None
    clip_recorder = ClipRecorder(
        enabled=config["clip_enabled"],
        pre_death_seconds=config["clip_pre_seconds"],
        post_death_seconds=config["clip_post_seconds"],
        output_dir=clip_output_dir,
        output_fps=config["clip_output_fps"],
    )

    bot = DeathBot(
        token=config["token"],
        prefix="!",
        channel=config["channel"],
        counter=counter,
        game=profile.display_name,
        clip_recorder=clip_recorder,
    )

    # Graceful shutdown
    stop_event = threading.Event()

    def shutdown(*_):
        logger.info("Shutting down...")
        stop_event.set()
        capture.stop()
        if audio_capture is not None:
            audio_capture.stop()
        summary = counter.end_session()
        logger.info(
            "Session summary: %d deaths this stream, %d total all-time",
            summary["deaths"],
            summary["total"],
        )

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start stream capture
    if not capture.start():
        logger.error(
            "Failed to start stream capture. Is %s live?", config["channel"]
        )
        logger.info("Starting bot in chat-only mode (no detection).")
        logger.info("Detection will start when the stream goes live.")

    # Start audio capture (separate lightweight stream connection)
    if audio_capture is not None and audio_detector is not None:
        if audio_capture.start(audio_detector.feed_audio):
            logger.info("Audio capture started")
        else:
            logger.warning("Audio capture failed to start — audio detection disabled")

    # Start detection in background thread
    loop = asyncio.new_event_loop()

    detection_thread = threading.Thread(
        target=detection_loop,
        args=(capture, detector, counter, clip_recorder, config["game"], bot, loop, stop_event),
        daemon=True,
    )
    detection_thread.start()

    # Run the bot (blocking)
    logger.info("Bot starting... Press Ctrl+C to stop.")
    try:
        bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
