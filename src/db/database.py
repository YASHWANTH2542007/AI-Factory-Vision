"""
SQLite persistence layer for inspection records.

Why SQLite instead of a bigger database (Postgres, MySQL)?
- Zero setup: it's a single file, no server process to run.
- Perfect for a single-machine factory inspection station.
- Easy to swap out later: this module is the ONLY place that talks to the
  database, so migrating to Postgres later only means changing this file.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection():
    """
    Context manager for DB connections. Guarantees the connection is closed
    even if an exception is raised mid-query -- a common source of leaked
    connections in beginner code.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't already exist. Safe to call every startup."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                object_id INTEGER,
                class_name TEXT NOT NULL,
                is_defective INTEGER NOT NULL,
                confidence REAL,
                width_mm REAL,
                height_mm REAL,
                source TEXT
            )
        """)
    logger.info("Database initialized")


def insert_inspection(class_name: str, is_defective: bool, confidence: float = None,
                       width_mm: float = None, height_mm: float = None,
                       object_id: int = None, source: str = "unknown"):
    """Insert a single inspection record."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO inspections
                (timestamp, object_id, class_name, is_defective, confidence, width_mm, height_mm, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            object_id,
            class_name,
            int(is_defective),
            confidence,
            width_mm,
            height_mm,
            source,
        ))


def get_all_inspections():
    """Return all inspection rows, most recent first."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM inspections ORDER BY timestamp DESC").fetchall()
        return [dict(row) for row in rows]


def get_summary_stats():
    """Aggregate counts used by the dashboard."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        defective = conn.execute("SELECT COUNT(*) FROM inspections WHERE is_defective = 1").fetchone()[0]
        good = total - defective
        return {
            "total": total,
            "good": good,
            "defective": defective,
            "defect_rate": round((defective / total) * 100, 2) if total else 0.0,
        }


def clear_all_inspections():
    """Wipe the table. Useful for demos/tests -- use with care in production."""
    with get_connection() as conn:
        conn.execute("DELETE FROM inspections")
    logger.info("All inspection records cleared")
