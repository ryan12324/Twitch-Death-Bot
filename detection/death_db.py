"""
SQLite-backed death database for the bot control panel.

Provides persistent, queryable death records with clip correlation
and the ability to manually adjust total deaths from the UI.

This layer sits alongside the existing JSON-based DeathCounter —
it does NOT replace it. The JSON system stays intact for main.py
CLI mode and bot/twitch_bot.py chat commands.
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


class DeathDatabase:
    """Thread-safe SQLite database for death tracking."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(DATA_DIR / "deaths.db")
        self._db_path = db_path
        self._lock = threading.Lock()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        logger.info("Death database initialized at %s", db_path)

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS deaths (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                session_count   INTEGER NOT NULL,
                total_count     INTEGER NOT NULL,
                confidence      REAL    NOT NULL,
                game            TEXT    NOT NULL,
                channel         TEXT    NOT NULL,
                clip_filename   TEXT,
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                game            TEXT    NOT NULL,
                channel         TEXT    NOT NULL,
                started_at      TEXT    NOT NULL,
                ended_at        TEXT,
                death_count     INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def record_death(
        self,
        timestamp: str,
        session_count: int,
        total_count: int,
        confidence: float,
        game: str,
        channel: str,
        clip_filename: str | None = None,
    ) -> int:
        """Insert a death record. Returns the new row id."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO deaths
                   (timestamp, session_count, total_count, confidence, game, channel, clip_filename)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (timestamp, session_count, total_count, confidence, game, channel, clip_filename),
            )
            self._conn.commit()
            return cur.lastrowid

    def start_session(self, game: str, channel: str) -> int:
        """Create a new session row. Returns the session id."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (game, channel, started_at) VALUES (?, ?, ?)",
                (game, channel, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def end_session(self, session_id: int, death_count: int) -> None:
        """Mark a session as ended."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ?, death_count = ? WHERE id = ?",
                (now, death_count, session_id),
            )
            self._conn.commit()

    def get_deaths(
        self, limit: int = 100, offset: int = 0, game: str | None = None
    ) -> list[dict]:
        """Return deaths newest-first."""
        if game:
            rows = self._conn.execute(
                "SELECT * FROM deaths WHERE game = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (game, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM deaths ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_death(self, death_id: int) -> dict | None:
        """Return a single death record or None."""
        row = self._conn.execute(
            "SELECT * FROM deaths WHERE id = ?", (death_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_death(self, death_id: int, **kwargs) -> bool:
        """Update allowed fields (notes, clip_filename) on a death record."""
        allowed = {"notes", "clip_filename"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [death_id]
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE deaths SET {set_clause} WHERE id = ?", values
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_death(self, death_id: int) -> bool:
        """Delete a death record. Returns True if a row was deleted."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM deaths WHERE id = ?", (death_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def get_total_deaths(self) -> int:
        """Return override total if set, otherwise COUNT(*)."""
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = 'total_deaths_override'"
        ).fetchone()
        if row:
            override = int(row["value"])
            if override > 0:
                return override
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM deaths").fetchone()
        return row["cnt"]

    def set_total_deaths(self, count: int) -> None:
        """Set a total deaths override. Pass 0 to revert to COUNT(*)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('total_deaths_override', ?)",
                (str(count),),
            )
            self._conn.commit()

    def get_sessions(self, limit: int = 50) -> list[dict]:
        """Return sessions newest-first."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self, game: str | None = None) -> dict:
        """Aggregate statistics."""
        total = self.get_total_deaths()

        if game:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM deaths WHERE game = ?", (game,)
            ).fetchone()
            game_deaths = row["cnt"]
        else:
            game_deaths = None

        # Per-game breakdown
        rows = self._conn.execute(
            "SELECT game, COUNT(*) AS cnt FROM deaths GROUP BY game ORDER BY cnt DESC"
        ).fetchall()
        per_game = {r["game"]: r["cnt"] for r in rows}

        session_count = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM sessions"
        ).fetchone()["cnt"]

        return {
            "total_deaths": total,
            "game_deaths": game_deaths,
            "per_game": per_game,
            "session_count": session_count,
        }
