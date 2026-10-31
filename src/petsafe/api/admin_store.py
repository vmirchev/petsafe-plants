"""Persistent admin settings and atomic rolling Meta API quota."""

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.getenv("PETSAFE_ADMIN_DB", "data/admin/admin.sqlite3"))


class MetaUnavailable(RuntimeError):
    """A paid Meta call was rejected before reaching Meta."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.execute("PRAGMA busy_timeout=30000")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS settings "
        "(id INTEGER PRIMARY KEY CHECK(id=1), "
        "enabled INTEGER NOT NULL, daily_limit INTEGER NOT NULL)"
    )
    db.execute("INSERT OR IGNORE INTO settings VALUES (1, 1, 100)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS calls "
        "(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, "
        "created REAL NOT NULL, cost REAL NOT NULL DEFAULT 0)"
    )
    db.execute("CREATE INDEX IF NOT EXISTS calls_created ON calls(created)")
    db.execute(
        "CREATE TABLE IF NOT EXISTS investigations "
        "(id INTEGER PRIMARY KEY, created REAL NOT NULL)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS investigations_created ON investigations(created)"
    )
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def record_investigation() -> None:
    """Record one accepted image investigation."""
    with _connect() as db:
        db.execute("INSERT INTO investigations(created) VALUES (?)", (time.time(),))


def reserve_call(kind: str) -> int:
    """Atomically reserve one paid call against the rolling 24-hour limit."""
    if kind not in {"spark", "sam"}:
        raise ValueError("Unknown Meta call type")
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        enabled, limit = db.execute(
            "SELECT enabled, daily_limit FROM settings WHERE id=1"
        ).fetchone()
        if not enabled:
            raise MetaUnavailable("api_disabled")
        now = time.time()
        used = db.execute(
            "SELECT COUNT(*) FROM calls WHERE created >= ?", (now - 86400,)
        ).fetchone()[0]
        if used >= limit:
            raise MetaUnavailable("quota_exceeded")
        cursor = db.execute(
            "INSERT INTO calls(kind, created) VALUES (?, ?)", (kind, now)
        )
        return cursor.lastrowid


def set_call_cost(call_id: int, cost: float) -> None:
    """Store the estimated cost after usage is available."""
    with _connect() as db:
        db.execute("UPDATE calls SET cost=? WHERE id=?", (max(0.0, cost), call_id))


def save_settings(enabled: bool, daily_limit: int) -> None:
    """Persist the admin kill switch and rolling call limit."""
    if not 0 <= daily_limit <= 1_000_000:
        raise ValueError("daily_limit must be between 0 and 1,000,000")
    with _connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "UPDATE settings SET enabled=?, daily_limit=? WHERE id=1",
            (int(enabled), daily_limit),
        )


def stats() -> dict:
    """Return rolling and lifetime counts from one consistent SQLite snapshot."""
    with _connect() as db:
        db.execute("BEGIN")
        enabled, limit = db.execute(
            "SELECT enabled, daily_limit FROM settings WHERE id=1"
        ).fetchone()
        cutoff = time.time() - 86400
        investigations = db.execute(
            "SELECT COUNT(*), SUM(created >= ?) FROM investigations", (cutoff,)
        ).fetchone()
        calls = db.execute(
            "SELECT kind, COUNT(*), SUM(created >= ?), SUM(cost), "
            "SUM(CASE WHEN created >= ? THEN cost ELSE 0 END) "
            "FROM calls GROUP BY kind",
            (cutoff, cutoff),
        ).fetchall()
    per_kind = {
        kind: {"total": total, "last_24h": recent or 0}
        for kind, total, recent, _, _ in calls
    }
    total_cost = sum(row[3] or 0 for row in calls)
    recent_cost = sum(row[4] or 0 for row in calls)
    used = sum(row[2] or 0 for row in calls)
    return {
        "enabled": bool(enabled),
        "daily_limit": limit,
        "remaining": max(0, limit - used),
        "investigations": {
            "total": investigations[0], "last_24h": investigations[1] or 0,
        },
        "spark": per_kind.get("spark", {"total": 0, "last_24h": 0}),
        "sam": per_kind.get("sam", {"total": 0, "last_24h": 0}),
        "estimated_cost_usd": {"total": total_cost, "last_24h": recent_cost},
    }
