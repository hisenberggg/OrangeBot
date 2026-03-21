"""Per-user chat threads and message transcripts in SQLite (chats.db)."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.app_data import ensure_data_dir


def _db_path() -> Path:
    return ensure_data_dir() / "chats.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_threads_user_updated
            ON chat_threads(user_id, updated_at DESC);
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            route TEXT,
            FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chat_messages_thread
            ON chat_messages(thread_id, id);
        """
    )


def _connect() -> sqlite3.Connection:
    ensure_data_dir()
    conn = sqlite3.connect(str(_db_path()))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _ensure_schema(conn)
    conn.commit()
    return conn


def clear_all_chats() -> None:
    """Delete every thread and message in chats.db. No-op if the database file is missing."""
    path = _db_path()
    if not path.exists():
        return
    with _connect() as conn:
        conn.execute("DELETE FROM chat_threads")
        conn.commit()


def list_threads(user_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM chat_threads
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
        }
        for r in rows
    ]


def create_thread(user_id: str, title: str = "New chat") -> str:
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    t = title[:200]
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_threads (id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, user_id, t, now, now),
        )
        conn.commit()
    return thread_id


def _ensure_thread(conn: sqlite3.Connection, user_id: str, thread_id: str) -> None:
    cur = conn.execute(
        "SELECT 1 FROM chat_threads WHERE id = ? AND user_id = ?",
        (thread_id, user_id),
    )
    if cur.fetchone():
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO chat_threads (id, user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (thread_id, user_id, "New chat", now, now),
    )


def get_messages(user_id: str, thread_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT 1 FROM chat_threads WHERE id = ? AND user_id = ?
            """,
            (thread_id, user_id),
        )
        if not cur.fetchone():
            return []
        cur = conn.execute(
            """
            SELECT role, content, timestamp, route
            FROM chat_messages
            WHERE thread_id = ?
            ORDER BY id
            """,
            (thread_id,),
        )
        rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for role, content, ts, route in rows:
        d: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": ts,
        }
        if route is not None:
            d["route"] = route
        out.append(d)
    return out


def assert_thread_owned(user_id: str, thread_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT 1 FROM chat_threads WHERE id = ? AND user_id = ?",
            (thread_id, user_id),
        )
        return cur.fetchone() is not None


def append_turn(
    user_id: str,
    thread_id: str,
    user_content: str,
    assistant_content: str,
    route: str | None = None,
) -> None:
    """Append user + assistant messages; bump thread updated_at and title if first turn."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        _ensure_thread(conn, user_id, thread_id)
        conn.execute(
            """
            INSERT INTO chat_messages (thread_id, role, content, timestamp, route)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, "user", user_content, now, None),
        )
        conn.execute(
            """
            INSERT INTO chat_messages (thread_id, role, content, timestamp, route)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, "assistant", assistant_content, now, route),
        )
        cur = conn.execute(
            "SELECT title FROM chat_threads WHERE id = ? AND user_id = ?",
            (thread_id, user_id),
        )
        row = cur.fetchone()
        new_title = row[0] if row else "New chat"
        if new_title in (None, "", "New chat") and user_content.strip():
            new_title = user_content.strip()[:200]
        conn.execute(
            """
            UPDATE chat_threads
            SET updated_at = ?, title = ?
            WHERE id = ? AND user_id = ?
            """,
            (now, new_title, thread_id, user_id),
        )
        conn.commit()
