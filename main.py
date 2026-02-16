"""
Twitch Death Counter Bot - CLI entry point (headless mode).

This is the alternative to the web UI (web/app.py). It reads all config
from .env, connects to a single Twitch channel, and runs detection until
you Ctrl+C. For most use cases, run the web app instead:

    python web/app.py       # then open http://localhost:4444/bot

The web UI lets you start/stop jobs, pick channels and profiles from
a browser, and see live detection video. This CLI mode is useful for
headless servers or if you only ever monitor one channel.
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
from bot.twitch_api import get_channel_game, match_profile_by_game_id
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
        "client_id": os.getenv("TWITCH_CLIENT_ID", ""),
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
        "command_prefix": os.getenv("COMMAND_PREFIX", "!"),
        "announce_deaths": os.getenv("ANNOUNCE_DEATHS", "true").lower() in ("true", "1", "yes"),
    }


def resolve_game_profile(config: dict) -> str:
    """Resolve the game profile to use.

    If GAME_PROFILE=auto and a TWITCH_CLIENT_ID is set, queries the
    Twitch Helix API to detect what game the channel is playing and
    matches it against known profiles. Falls back to 'generic'.
    """
    game = config["game"]

    if game == "auto":
        client_id = config["client_id"]
        if not client_id:
            logger.warning(
                "GAME_PROFILE=auto requires TWITCH_CLIENT_ID to be set. "
                "Falling back to 'generic' profile."
            )
            return "generic"

        logger.info("Auto-detecting game for channel %s...", config["channel"])
        info = get_channel_game(config["channel"], client_id, config["token"])

        if info:
            logger.info(
                "Channel is playing: %s (game_id=%s)",
                info["game_name"],
                info["game_id"],
            )
            matched = match_profile_by_game_id(info["game_id"])
            if matched:
                logger.info("Matched game profile: %s", matched)
                return matched
            else:
                logger.info(
                    "No specific profile for '%s', using 'generic'.",
                    info["game_name"],
                )
                return "generic"
        else:
            logger.warning(
                "Could not detect game (channel may be offline). Using 'generic'."
            )
            return "generic"

    # Validate the specified profile exists
    if game not in list_profiles():
        logger.error(
            "Unknown GAME_PROFILE '%s'. Available: %s",
            game,
            ", ".join(list_profiles()),
        )
        sys.exit(1)

    return game


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

            # Schedule chat announcement on the bot's event loop.
            # bot.loop is the asyncio loop that twitchio runs on, which
            # is the only loop where bot coroutines can be awaited.
            try:
                loop = bot.loop
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        bot.announce_death(session_count, total_count),
                        loop,
                    )
                else:
                    logger.warning(
                        "Bot event loop not running — skipping chat announcement "
                        "for death #%d",
                        session_count,
                    )
            except Exception as exc:
                logger.error("Failed to schedule death announcement: %s", exc)

    # Flush any in-progress clip on shutdown
    clip_recorder.flush()
    logger.info("Detection loop stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = get_config()

    game_name = resolve_game_profile(config)
    profile = get_profile(game_name)
    logger.info("Game profile: %s", profile.display_name)

    # Initialize components
    counter = DeathCounter()
    counter.start_session(game_name)

    detector = DeathDetector(
        profile=profile,
        threshold=config["threshold"],
        cooldown=config["cooldown"],
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
        prefix=config["command_prefix"],
        channel=config["channel"],
        counter=counter,
        game=profile.display_name,
        clip_recorder=clip_recorder,
        announce_enabled=config["announce_deaths"],
    )

    # Graceful shutdown
    stop_event = threading.Event()
    shutdown_called = threading.Event()

    def shutdown(*_):
        if shutdown_called.is_set():
            return
        shutdown_called.set()
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

    # Start detection in background thread.
    # The detection thread uses bot.loop (twitchio's event loop) to schedule
    # chat announcements via asyncio.run_coroutine_threadsafe.
    detection_thread = threading.Thread(
        target=detection_loop,
        args=(capture, detector, counter, clip_recorder, game_name, bot, stop_event),
        daemon=True,
    )
    detection_thread.start()

    # Run the bot (blocking — runs twitchio's asyncio event loop)
    logger.info("Bot starting... Press Ctrl+C to stop.")
    try:
        bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
