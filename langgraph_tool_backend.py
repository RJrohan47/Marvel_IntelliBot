"""Backward-compatible public entry point for Marvel's LangGraph chatbot.

The workflow lives in ``marvel_workflow.py`` so state, nodes, and routes can
be reviewed independently from persistence and the Streamlit UI.
"""

import sqlite3
import logging
from pathlib import Path
from threading import Lock
from time import monotonic

from langgraph.checkpoint.sqlite import SqliteSaver

from marvel_workflow import build_workflow, generate_chat_title
from thread_manager import delete_thread_record, find_expired_threads
from youtube_rag.service import clear_thread_video


DATABASE_PATH = Path(__file__).with_name("marvel_chatbot.db")
connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)
chatbot = build_workflow(checkpointer)
logger = logging.getLogger(__name__)

THREAD_RETENTION_DAYS = 45
CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60
_cleanup_lock = Lock()
_last_cleanup_at: float | None = None


def delete_thread_memory(thread_id: str) -> None:
    """Delete every LangGraph checkpoint and pending write for one chat thread."""

    checkpointer.delete_thread(str(thread_id))


def cleanup_expired_threads() -> list[str]:
    """Delete inactive threads, checking at most once per day per app process."""
    global _last_cleanup_at

    with _cleanup_lock:
        now = monotonic()
        if _last_cleanup_at is not None and now - _last_cleanup_at < CLEANUP_INTERVAL_SECONDS:
            return []

        expired_threads = find_expired_threads(THREAD_RETENTION_DAYS)
        deleted_threads = []
        for thread_id in expired_threads:
            try:
                delete_thread_memory(thread_id)
                clear_thread_video(thread_id)
                delete_thread_record(thread_id)
                deleted_threads.append(thread_id)
            except Exception:
                logger.exception("Could not fully remove expired thread %s", thread_id)

        _last_cleanup_at = now
        if deleted_threads:
            logger.info("Removed %s inactive chat threads", len(deleted_threads))
        return deleted_threads
