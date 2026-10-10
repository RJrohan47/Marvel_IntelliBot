from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


DATABASE_PATH = Path(__file__).with_name("marvel_chatbot.db")


class ChatThread(TypedDict):
    """The metadata needed to render one entry in the chat sidebar."""

    thread_id: str
    title: str
   

def _connection() -> sqlite3.Connection:
    """Create a short-lived connection for a metadata operation."""

    return sqlite3.connect(DATABASE_PATH)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def initialize_thread_table() -> None:
    """Create the application-owned thread metadata table when needed."""

    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_threads (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def save_thread_title(thread_id: str, title: str) -> None:
    """Create or update a sidebar title for a LangGraph thread."""

    clean_title = " ".join(title.split())[:80] or "New conversation"
    now = _timestamp()

    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_threads (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title = excluded.title,
                updated_at = excluded.updated_at
            """,
            (str(thread_id), clean_title, now, now),
        )


def touch_thread(thread_id: str) -> None:
    """Update a thread's ordering timestamp after a new message is sent."""

    with _connection() as connection:
        connection.execute(
            """
            UPDATE chat_threads
            SET updated_at = ?
            WHERE thread_id = ?
            """,
            (_timestamp(), str(thread_id)),
        )


def delete_thread_record(thread_id: str) -> None:
    """Remove the thread from the sidebar metadata table."""

    with _connection() as connection:
        connection.execute(
            "DELETE FROM chat_threads WHERE thread_id = ?",
            (str(thread_id),),
        )



def fetch_thread_list() -> list[dict[str, str]]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT thread_id, title
            FROM chat_threads
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [
        {
            "thread_id": row[0],
            "title": row[1],
        }
        for row in rows
    ]
