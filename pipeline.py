"""Ingestion logic shared by the Inngest functions, the batch script and the evaluation."""
from __future__ import annotations

import uuid

from custom_types import Chunk, RAGUpsertResult
from retrieval import EmbedFn
from vector_db import QdrantStorage


def chunk_ids(source_id: str, n: int) -> list[str]:
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(n)]


def index_chunks(source_id: str, chunks: list[Chunk], store: QdrantStorage, embed_fn: EmbedFn) -> RAGUpsertResult:
    """Replace a document's chunks in the store.

    Old chunks are deleted first, so re-ingesting a shorter version of a document does not
    leave stale chunks behind.
    """
    replaced = store.delete_source(source_id)
    if not chunks:
        return RAGUpsertResult(ingested=0, replaced=replaced)
    vectors = embed_fn([c.text for c in chunks])
    payloads = [
        {"source": source_id, "text": c.text, "page": c.page, "chunk_index": i}
        for i, c in enumerate(chunks)
    ]
    store.upsert(chunk_ids(source_id, len(chunks)), vectors, payloads)
    return RAGUpsertResult(ingested=len(chunks), replaced=replaced)
