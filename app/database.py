import sqlite3
import os
import logging
from config import settings

logger = logging.getLogger(__name__)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    os.makedirs(os.path.dirname(settings.database_path), exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            department  TEXT    DEFAULT '',
            employee_id TEXT    DEFAULT '',
            is_active   INTEGER DEFAULT 1,
            created_at  TEXT    DEFAULT (datetime('now', 'localtime')),
            updated_at  TEXT    DEFAULT (datetime('now', 'localtime'))
        );

        CREATE TABLE IF NOT EXISTS face_embeddings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id  INTEGER NOT NULL,
            embedding  BLOB    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (person_id) REFERENCES persons(id)
        );

        CREATE TABLE IF NOT EXISTS events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id        INTEGER,
            person_name      TEXT    DEFAULT '未知',
            confidence       REAL    DEFAULT 0.0,
            status           TEXT    DEFAULT 'detected',
            entry_reason     TEXT    DEFAULT '',
            transcript       TEXT    DEFAULT '',
            summary          TEXT    DEFAULT '',
            event_image_path TEXT    DEFAULT '',
            created_at       TEXT    DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (person_id) REFERENCES persons(id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
        CREATE INDEX IF NOT EXISTS idx_events_person_id  ON events(person_id);
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", settings.database_path)


def cleanup_old_events():
    conn = get_db()
    cur = conn.execute(
        "DELETE FROM events WHERE created_at < datetime('now', 'localtime', ?)",
        (f"-{settings.retention_days} days",),
    )
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    if deleted:
        logger.info("Cleaned up %d old event(s) (retention=%d days)", deleted, settings.retention_days)
