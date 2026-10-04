import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embeddings import HashEmbedder  # noqa: E402
from vector_db import QdrantStorage  # noqa: E402


@pytest.fixture
def embed():
    return HashEmbedder(dim=64)


@pytest.fixture
def store(request):
    # A fresh in-memory collection per test
    return QdrantStorage(location=":memory:", collection=f"t_{request.node.name}", dim=64)


@pytest.fixture
def sample_docs():
    return ROOT / "eval" / "sample_docs"
