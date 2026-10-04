from __future__ import annotations

import pydantic


class Chunk(pydantic.BaseModel):
    """A piece of a document, with the page it came from (None for formats without pages)."""
    text: str
    page: int | None = None


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[Chunk]
    source_id: str


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int
    replaced: int = 0


class RetrievedChunk(pydantic.BaseModel):
    text: str
    source: str
    page: int | None = None
    chunk_index: int | None = None
    score: float = 0.0


class RAGSearchResult(pydantic.BaseModel):
    contexts: list[RetrievedChunk]

    @property
    def sources(self) -> list[str]:
        # Keep retrieval order, drop duplicates
        return list(dict.fromkeys(c.source for c in self.contexts))


class Citation(pydantic.BaseModel):
    ref: int
    source: str
    page: int | None = None
    snippet: str


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    standalone_question: str
    sources: list[str]
    citations: list[Citation]
    num_contexts: int
    retrieval_mode: str
