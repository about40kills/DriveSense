"""
DriveSense — SQLite event logger.

Persists detection events (DROWSY, YAWNING, DISTRACTED, MICRO-SLEEP) to
data/events.db so they survive application restarts.

Usage
-----
    import db
    db.init_db()                  # call once at startup
    db.log_event("DROWSY", avg_ear=0.31, mouth_ratio=0.08)
    rows = db.get_recent_events(50)   # list of dicts, newest first
"""
import os
import sqlite3
import time

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, "data", "events.db")


def _connect() -> sqlite3.Connection:
    """Open a new connection (safe to call from any thread)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the events table if it doesn't exist."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL    NOT NULL,
                event       TEXT    NOT NULL,
                avg_ear     REAL    DEFAULT 0.0,
                mouth_ratio REAL    DEFAULT 0.0
            )
        """)
        conn.commit()


def log_event(event: str, avg_ear: float = 0.0, mouth_ratio: float = 0.0) -> None:
    """Insert one event row.  Called from the detection thread — fast and safe."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (ts, event, avg_ear, mouth_ratio) VALUES (?, ?, ?, ?)",
            (time.time(), event, round(avg_ear, 4), round(mouth_ratio, 4)),
        )
        conn.commit()


def get_recent_events(limit: int = 50) -> list[dict]:
    """Return the *limit* most recent events, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT ts, event, avg_ear, mouth_ratio
               FROM events ORDER BY ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_event_counts(window_seconds: int = 3600) -> dict:
    """
    Count events in the last *window_seconds* seconds, grouped by event type.
    Useful for per-session summary stats on the dashboard.
    """
    cutoff = time.time() - window_seconds
    with _connect() as conn:
        rows = conn.execute(
            """SELECT event, COUNT(*) AS cnt
               FROM events WHERE ts >= ?
               GROUP BY event""",
            (cutoff,),
        ).fetchall()
    return {r["event"]: r["cnt"] for r in rows}
