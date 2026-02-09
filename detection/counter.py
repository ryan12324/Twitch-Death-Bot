"""
Death counter with persistent JSON storage.

Tracks:
- Deaths this stream session
- Total deaths all-time
- Per-game death counts
- Death history with timestamps
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEATHS_FILE = DATA_DIR / "deaths.json"


def _default_data() -> dict:
    return {
        "total_deaths": 0,
        "games": {},
        "sessions": [],
        "current_session": None,
    }


class DeathCounter:
    def __init__(self):
        self.data: dict = _default_data()
        self.session_deaths: int = 0
        self.session_start: str = ""
        self._load()

    def _load(self) -> None:
        """Load persistent death data from disk."""
        if DEATHS_FILE.exists():
            try:
                with open(DEATHS_FILE, "r") as f:
                    self.data = json.load(f)
                logger.info(
                    "Loaded death data: %d total deaths",
                    self.data.get("total_deaths", 0),
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load deaths file: %s", e)
                self.data = _default_data()
        else:
            self.data = _default_data()

    def _save(self) -> None:
        """Persist death data to disk."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(DEATHS_FILE, "w") as f:
                json.dump(self.data, f, indent=2)
        except OSError as e:
            logger.error("Failed to save deaths file: %s", e)

    def start_session(self, game: str) -> None:
        """Begin a new stream session."""
        self.session_deaths = 0
        self.session_start = datetime.now(timezone.utc).isoformat()
        self.data["current_session"] = {
            "game": game,
            "start": self.session_start,
            "deaths": 0,
        }
        self._save()
        logger.info("Session started for %s", game)

    def record_death(self, game: str, confidence: float = 0.0) -> int:
        """
        Record a death. Returns the new session death count.
        """
        self.session_deaths += 1
        self.data["total_deaths"] = self.data.get("total_deaths", 0) + 1

        # Per-game tracking
        if game not in self.data.get("games", {}):
            self.data.setdefault("games", {})[game] = 0
        self.data["games"][game] += 1

        # Update current session
        if self.data.get("current_session"):
            self.data["current_session"]["deaths"] = self.session_deaths

        self._save()

        logger.info(
            "Death #%d this session (#%d total) [%s] confidence=%.2f",
            self.session_deaths,
            self.data["total_deaths"],
            game,
            confidence,
        )

        return self.session_deaths

    def end_session(self) -> dict:
        """End the current session and archive it. Returns session summary."""
        session = self.data.get("current_session")
        summary = {
            "deaths": self.session_deaths,
            "total": self.data.get("total_deaths", 0),
        }

        if session:
            session["end"] = datetime.now(timezone.utc).isoformat()
            session["deaths"] = self.session_deaths
            self.data.setdefault("sessions", []).append(session)
            self.data["current_session"] = None

        self._save()
        logger.info("Session ended: %d deaths", self.session_deaths)
        return summary

    @property
    def total_deaths(self) -> int:
        return self.data.get("total_deaths", 0)

    def get_game_deaths(self, game: str) -> int:
        return self.data.get("games", {}).get(game, 0)

    def get_stats(self, game: str) -> dict:
        """Get a summary of death statistics."""
        return {
            "session_deaths": self.session_deaths,
            "total_deaths": self.total_deaths,
            "game_deaths": self.get_game_deaths(game),
            "game": game,
            "sessions_played": len(self.data.get("sessions", [])),
        }
