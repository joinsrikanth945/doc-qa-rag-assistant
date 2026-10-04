"""Embedding back-ends.

`embed_texts` calls OpenAI in batches. `HashEmbedder` is a deterministic, dependency-free
bag-of-words embedder used by the tests and by `eval/run_eval.py --offline`, so the
pipeline can be exercised without an API key.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI  # imported lazily so tests don't need a key

    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with OpenAI, batching to stay under request size limits."""
    if not texts:
        return []
    client = _openai_client()
    vectors: list[list[float]] = []
    size = settings.embed_batch_size
    for start in range(0, len(texts), size):
        batch = texts[start:start + size]
        response = client.embeddings.create(model=settings.embed_model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Feature-hashing embedder: cheap, deterministic, good enough for tests."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
