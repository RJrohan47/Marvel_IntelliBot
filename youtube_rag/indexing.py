"""Stage 4: embed chunks and persist a FAISS index per video."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from youtube_rag.config import EMBEDDING_MODEL, INDEX_ROOT


def build_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for YouTube embeddings.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)


class FaissVideoIndex:
    def __init__(self, embeddings: GoogleGenerativeAIEmbeddings) -> None:
        self.embeddings = embeddings

    def load_or_build(self, video_id: str, chunks: list[Document] | None = None) -> FAISS:
        path = self._path(video_id)
        if self.exists(video_id):
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        if not chunks:
            raise RuntimeError("Chunks are required to build a new video index.")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(path)
        return vector_store

    def exists(self, video_id: str) -> bool:
        path = self._path(video_id)
        return (path / "index.faiss").exists() and (path / "index.pkl").exists()

    @staticmethod
    def _path(video_id: str) -> Path:
        return INDEX_ROOT / video_id
