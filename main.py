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

    return {
        "token": token,
        "channel": channel,
        "game": game,
        "capture_interval": float(os.getenv("CAPTURE_INTERVAL", "2.0")),
        "threshold": float(os.getenv("DETECTION_THRESHOLD", "0.80")),
        "cooldown": int(os.getenv("DEATH_COOLDOWN", "15")),
        "quality": os.getenv("STREAM_QUALITY", "720p"),
    }


# ---------------------------------------------------------------------------
# Detection loop (runs in a background thread)
# ---------------------------------------------------------------------------


def detection_loop(
    capture: StreamCapture,
    detector: DeathDetector,
    counter: DeathCounter,
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
            stop_event.wait(0.5)
            continue

        is_death, confidence = detector.analyze_frame(frame)

        if is_death:
            session_count = counter.record_death(game, confidence)
            total_count = counter.total_deaths

            # Schedule chat announcement on the bot's event loop
            asyncio.run_coroutine_threadsafe(
                bot.announce_death(session_count, total_count),
                loop,
            )

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

    detector = DeathDetector(
        profile=profile,
        threshold=config["threshold"],
        cooldown=config["cooldown"],
    )

    capture = StreamCapture(
        channel=config["channel"],
        quality=config["quality"],
        capture_interval=config["capture_interval"],
    )

    bot = DeathBot(
        token=config["token"],
        prefix="!",
        channel=config["channel"],
        counter=counter,
        game=profile.display_name,
    )

    # Graceful shutdown
    stop_event = threading.Event()

    def shutdown(*_):
        logger.info("Shutting down...")
        stop_event.set()
        capture.stop()
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

    # Start detection in background thread
    loop = asyncio.new_event_loop()

    detection_thread = threading.Thread(
        target=detection_loop,
        args=(capture, detector, counter, config["game"], bot, loop, stop_event),
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
