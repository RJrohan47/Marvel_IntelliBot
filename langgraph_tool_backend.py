"""Backward-compatible public entry point for Marvel's LangGraph chatbot.

The workflow lives in ``marvel_workflow.py`` so state, nodes, and routes can
be reviewed independently from persistence and the Streamlit UI.
"""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from marvel_workflow import build_workflow, generate_chat_title


DATABASE_PATH = Path(__file__).with_name("marvel_chatbot.db")
connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)
chatbot = build_workflow(checkpointer)


def delete_thread_memory(thread_id: str) -> None:
    """Delete every LangGraph checkpoint and pending write for one chat thread."""

    checkpointer.delete_thread(str(thread_id))
