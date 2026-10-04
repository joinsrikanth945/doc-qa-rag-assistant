"""Thin wrapper around Qdrant: upsert, dense search, per-source management."""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings
from custom_types import RetrievedChunk

_clients: dict[str, QdrantClient] = {}


def _get_client(location: str | None = None) -> QdrantClient:
    """Reuse one client per location: local on-disk Qdrant only allows a single open client."""
    key = location or settings.qdrant_url or settings.qdrant_path
    if key not in _clients:
        if key == ":memory:":
            _clients[key] = QdrantClient(location=":memory:")
        elif key.startswith("http"):
            _clients[key] = QdrantClient(url=key, api_key=settings.qdrant_api_key)
        else:
            _clients[key] = QdrantClient(path=key)
    return _clients[key]


def _source_filter(source_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_id))])


class QdrantStorage:
    def __init__(self, location: str | None = None, collection: str | None = None, dim: int | None = None):
        self.client = _get_client(location)
        self.collection = collection or settings.collection
        dim = dim or settings.embed_dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads) -> None:
        points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads)]
        self.client.upsert(self.collection, points=points)

    def delete_source(self, source_id: str) -> int:
        """Remove every chunk of a document. Returns how many chunks were removed."""
        existing = self.client.count(self.collection, count_filter=_source_filter(source_id), exact=True).count
        if existing:
            self.client.delete(self.collection, points_selector=_source_filter(source_id))
        return existing

    def list_sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for payload in self._iter_payloads():
            src = payload.get("source", "")
            counts[src] = counts.get(src, 0) + 1
        return dict(sorted(counts.items()))

    def all_chunks(self) -> list[RetrievedChunk]:
        """Every stored chunk; used to build the BM25 keyword index."""
        return [self._to_chunk(p) for p in self._iter_payloads() if p.get("text")]

    def dense_search(self, query_vector, top_k: int = 5) -> list[RetrievedChunk]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            if payload.get("text"):
                results.append(self._to_chunk(payload, score=point.score))
        return results

    # Backwards-compatible shape used by the original code
    def search(self, query_vector, top_k: int = 5) -> dict:
        found = self.dense_search(query_vector, top_k)
        return {
            "contexts": [c.text for c in found],
            "sources": list(dict.fromkeys(c.source for c in found)),
        }

    def _iter_payloads(self):
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, with_payload=True, with_vectors=False, limit=256, offset=offset
            )
            for point in points:
                yield point.payload or {}
            if offset is None:
                break

    @staticmethod
    def _to_chunk(payload: dict, score: float = 0.0) -> RetrievedChunk:
        return RetrievedChunk(
            text=payload.get("text", ""),
            source=payload.get("source", ""),
            page=payload.get("page"),
            chunk_index=payload.get("chunk_index"),
            score=score,
        )
