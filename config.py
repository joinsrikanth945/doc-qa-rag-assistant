"""Central configuration, read once from environment variables (and .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Models
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-5.6-luna")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-large")
    embed_dim: int = int(os.getenv("EMBED_DIM", "3072"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "96"))

    # Vector store: set QDRANT_URL to use a Qdrant server, otherwise a local on-disk DB is used
    qdrant_url: str | None = os.getenv("QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_path: str = os.getenv("QDRANT_PATH", "./qdrant_local_db")
    collection: str = os.getenv("QDRANT_COLLECTION", "docs")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")  # "dense" or "hybrid"
    candidate_pool: int = int(os.getenv("CANDIDATE_POOL", "20"))  # candidates per retriever before fusion
    rerank: bool = _bool("RERANK", False)
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))


settings = Settings()

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")
