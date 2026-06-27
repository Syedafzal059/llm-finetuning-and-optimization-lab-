"""SQLite usage tracking per client project."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

from core.utils.config_loader import get_project_dir

USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp    TEXT NOT NULL,
  tokens_in    INTEGER NOT NULL,
  tokens_out   INTEGER NOT NULL,
  latency_ms   INTEGER NOT NULL,
  status       TEXT NOT NULL,
  error_msg    TEXT
);
"""


class UsageTracker:
    """Per-project SQLite usage logger and query helper."""

    def __init__(self, project_name: str) -> None:
        """Initialize tracker and ensure the database schema exists."""
        self._project_name = project_name
        self._db_path = get_project_dir({"project_name": project_name}) / "usage.db"
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        """Return the absolute path to the usage database."""
        return self._db_path

    def _ensure_schema(self) -> None:
        """Create the usage table if it does not exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(USAGE_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a SQLite connection with row factory enabled."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _current_month_prefix() -> str:
        """Return YYYY-MM prefix for the current UTC month."""
        return datetime.now(timezone.utc).strftime("%Y-%m")

    @staticmethod
    def _today_prefix() -> str:
        """Return YYYY-MM-DD prefix for the current UTC day."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def log_request(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        status: str,
        error_msg: Optional[str] = None,
    ) -> None:
        """Record a request outcome in the usage database."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO usage (timestamp, tokens_in, tokens_out, latency_ms, status, error_msg)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    tokens_in,
                    tokens_out,
                    int(round(latency_ms)),
                    status,
                    error_msg,
                ),
            )
            conn.commit()

    def get_today_count(self) -> int:
        """Return the number of successful requests logged today (UTC)."""
        prefix = self._today_prefix()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM usage WHERE timestamp LIKE ? AND status = 'success'",
                (f"{prefix}%",),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_month_count(self) -> int:
        """Return the number of successful requests logged this month (UTC)."""
        prefix = self._current_month_prefix()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM usage WHERE timestamp LIKE ? AND status = 'success'",
                (f"{prefix}%",),
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_month_tokens(self) -> int:
        """Return total tokens (in + out) consumed this month (UTC)."""
        prefix = self._current_month_prefix()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(tokens_in + tokens_out), 0) AS total
                FROM usage
                WHERE timestamp LIKE ? AND status = 'success'
                """,
                (f"{prefix}%",),
            ).fetchone()
        return int(row["total"]) if row else 0

    def get_avg_latency(self) -> float:
        """Return average latency in milliseconds for successful requests this month."""
        prefix = self._current_month_prefix()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT AVG(latency_ms) AS avg_ms
                FROM usage
                WHERE timestamp LIKE ? AND status = 'success'
                """,
                (f"{prefix}%",),
            ).fetchone()
        if row and row["avg_ms"] is not None:
            return round(float(row["avg_ms"]), 2)
        return 0.0

    def get_rpm(self) -> float:
        """Return requests per minute over the last 60 seconds."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=60)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM usage
                WHERE timestamp >= ? AND status = 'success'
                """,
                (cutoff,),
            ).fetchone()
        return round(float(row["cnt"]), 1) if row else 0.0
