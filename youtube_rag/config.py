import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "marvel_chatbot.db"
INDEX_ROOT = PROJECT_ROOT / "data" / "youtube_indexes"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
CHUNK_SIZE = int(os.getenv("YOUTUBE_RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("YOUTUBE_RAG_CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("YOUTUBE_RAG_RETRIEVAL_K", "5"))
MMR_FETCH_K = int(os.getenv("YOUTUBE_RAG_MMR_FETCH_K", "20"))
