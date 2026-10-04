"""Hybrid retrieval: dense vectors + BM25 keywords, merged with Reciprocal Rank Fusion.

Dense search is good at paraphrases ("how do I restart it?"), while BM25 is good at exact
terms such as error codes, model numbers and product names that embeddings often blur.
Fusing both rankings gives more robust results than either alone.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from rank_bm25 import BM25Plus

from custom_types import RetrievedChunk
from vector_db import QdrantStorage

EmbedFn = Callable[[list[str]], list[list[float]]]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]*")
_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or the this to was what when where which who why with "
    "does do can my your you".split()
)


def tokenize(text: str) -> list[str]:
    tokens = (t.strip(".-_") for t in _TOKEN_RE.findall(text.lower()))
    return [t for t in tokens if t and t not in _STOPWORDS]


def _key(chunk: RetrievedChunk) -> tuple:
    return (chunk.source, chunk.chunk_index, chunk.text[:80])


def bm25_search(question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    query = tokenize(question)
    if not query:
        return []
    corpus = [tokenize(c.text) for c in chunks]
    # BM25+ keeps term weights positive even in tiny corpora (plain BM25 can give 0 or negative IDF)
    scores = BM25Plus(corpus).get_scores(query)
    query_terms = set(query)
    ranked = sorted(
        ((c, s) for c, s, toks in zip(chunks, scores, corpus) if query_terms & set(toks)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [c.model_copy(update={"score": float(s)}) for c, s in ranked[:top_k]]


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]], k: int = 60) -> list[RetrievedChunk]:
    """Combine several ranked lists: score = sum over lists of 1 / (k + rank)."""
    fused: dict[tuple, float] = {}
    first_seen: dict[tuple, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            key = _key(chunk)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, chunk)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [first_seen[key].model_copy(update={"score": score}) for key, score in ordered]


def retrieve(
    question: str,
    store: QdrantStorage,
    embed_fn: EmbedFn,
    top_k: int = 5,
    mode: str = "hybrid",
    candidate_pool: int = 20,
) -> list[RetrievedChunk]:
    pool = max(candidate_pool, top_k)
    query_vec = embed_fn([question])[0]
    dense = store.dense_search(query_vec, pool)
    if mode == "dense":
        return dense[:top_k]
    if mode != "hybrid":
        raise ValueError(f"Unknown retrieval mode '{mode}' (use 'dense' or 'hybrid')")
    keyword = bm25_search(question, store.all_chunks(), pool)
    return reciprocal_rank_fusion([dense, keyword])[:top_k]


# ---------- LLM re-ranking ----------

def build_rerank_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    passages = "\n\n".join(f"[{i}] {c.text[:700]}" for i, c in enumerate(chunks))
    return [
        {
            "role": "system",
            "content": "You rank passages by how useful they are for answering a question. "
                       'Reply with JSON only: {"ranking": [passage numbers, most useful first]}.',
        },
        {"role": "user", "content": f"Question: {question}\n\nPassages:\n{passages}"},
    ]


def apply_rerank(chunks: list[RetrievedChunk], llm_reply: str, top_k: int) -> list[RetrievedChunk]:
    """Reorder chunks using the LLM's ranking; fall back to the original order on bad output."""
    try:
        match = re.search(r"\{.*\}", llm_reply, re.DOTALL)
        order = json.loads(match.group(0))["ranking"] if match else []
    except (ValueError, KeyError, TypeError):
        order = []
    seen: set[int] = set()
    ranked = []
    for idx in order:
        if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            ranked.append(chunks[idx])
    ranked.extend(c for i, c in enumerate(chunks) if i not in seen)
    return ranked[:top_k]
