"""YouTube RAG orchestration and thread-to-video persistence.

Pipeline: load/translate → clean → sentence-aware split → embed/FAISS → MMR → generate.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from youtube_rag.cleaning import TranscriptCleaner
from youtube_rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DATABASE_PATH, EMBEDDING_MODEL, MMR_FETCH_K, RETRIEVAL_K
from youtube_rag.generation import GroundedAnswerGenerator
from youtube_rag.indexing import FaissVideoIndex, build_embeddings
from youtube_rag.loader import YouTubeTranscriptLoader, extract_video_id
from youtube_rag.retrieval import MMRTranscriptRetriever
from youtube_rag.splitter import SentenceAwareTranscriptSplitter


def _initialize_tables() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS youtube_videos (
                video_id TEXT PRIMARY KEY, canonical_url TEXT NOT NULL,
                embedding_model TEXT NOT NULL, chunk_size INTEGER NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS thread_videos (
                thread_id TEXT PRIMARY KEY, video_id TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def _save_video_metadata(video_id: str) -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO youtube_videos(video_id, canonical_url, embedding_model, chunk_size)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                canonical_url = excluded.canonical_url,
                embedding_model = excluded.embedding_model,
                chunk_size = excluded.chunk_size,
                indexed_at = CURRENT_TIMESTAMP
        """, (video_id, f"https://www.youtube.com/watch?v={video_id}", EMBEDDING_MODEL, CHUNK_SIZE))


@lru_cache(maxsize=8)
def get_or_create_vector_store(video_id: str) -> FAISS:
    """Build a video index once, then reuse its persisted FAISS representation."""
    index = FaissVideoIndex(build_embeddings())
    if index.exists(video_id):
        return index.load_or_build(video_id)

    raw_transcript = YouTubeTranscriptLoader().load(video_id)
    cleaned_transcript = TranscriptCleaner().clean(raw_transcript)
    chunks = SentenceAwareTranscriptSplitter(CHUNK_SIZE, CHUNK_OVERLAP).split(cleaned_transcript)
    vector_store = index.load_or_build(video_id, chunks)
    _save_video_metadata(video_id)
    return vector_store


def prepare_video(video_or_url: str) -> str:
    _initialize_tables()
    video_id = extract_video_id(video_or_url)
    get_or_create_vector_store(video_id)
    return video_id


def set_thread_video(thread_id: str, video_or_url: str) -> str:
    video_id = prepare_video(video_or_url)
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO thread_videos(thread_id, video_id, canonical_url)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                video_id = excluded.video_id,
                canonical_url = excluded.canonical_url,
                updated_at = CURRENT_TIMESTAMP
        """, (thread_id, video_id, canonical_url))
    return video_id


def get_thread_video(thread_id: str) -> dict[str, str] | None:
    _initialize_tables()
    with sqlite3.connect(DATABASE_PATH) as connection:
        row = connection.execute("SELECT video_id, canonical_url FROM thread_videos WHERE thread_id = ?", (thread_id,)).fetchone()
    return {"video_id": row[0], "canonical_url": row[1]} if row else None


def clear_thread_video(thread_id: str) -> None:
    _initialize_tables()
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("DELETE FROM thread_videos WHERE thread_id = ?", (thread_id,))


def answer_video_question(question: str, video_id: str) -> str:
    chunks = retrieve_video_chunks(question, video_id)
    return generate_video_answer(question, video_id, chunks)


def retrieve_video_chunks(question: str, video_id: str) -> list[Document]:
    """Retrieve transcript evidence from the cached index for a video question."""
    vector_store = get_or_create_vector_store(video_id)
    return MMRTranscriptRetriever(
        vector_store, top_k=RETRIEVAL_K, fetch_k=MMR_FETCH_K
    ).retrieve(question)


@lru_cache(maxsize=1)
def _get_answer_generator() -> GroundedAnswerGenerator:
    """Reuse the configured answer model across questions in this process."""
    return GroundedAnswerGenerator()


def generate_video_answer(
    question: str,
    video_id: str,
    chunks: list[Document],
    user_memory_context: str = "",
) -> str:
    """Generate a grounded response from retrieved transcript evidence."""
    return _get_answer_generator().generate(
        question, video_id, chunks, user_memory_context
    )
