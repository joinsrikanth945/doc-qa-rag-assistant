from custom_types import Chunk
from pipeline import chunk_ids, index_chunks


def test_reingest_replaces_old_chunks(store, embed):
    first = index_chunks("doc.md", [Chunk(text=f"part {i}") for i in range(5)], store, embed)
    assert first.ingested == 5 and first.replaced == 0

    # A shorter new version must not leave chunks 2-4 of the old version behind
    second = index_chunks("doc.md", [Chunk(text="new a"), Chunk(text="new b")], store, embed)
    assert second.replaced == 5
    assert store.list_sources() == {"doc.md": 2}
    assert {c.text for c in store.all_chunks()} == {"new a", "new b"}


def test_delete_source_only_touches_that_source(store, embed):
    index_chunks("a.md", [Chunk(text="alpha")], store, embed)
    index_chunks("b.md", [Chunk(text="beta"), Chunk(text="gamma")], store, embed)
    assert store.delete_source("b.md") == 2
    assert store.list_sources() == {"a.md": 1}
    assert store.delete_source("missing.md") == 0


def test_payload_keeps_page_and_index(store, embed):
    index_chunks("p.pdf", [Chunk(text="page three text", page=3)], store, embed)
    (chunk,) = store.all_chunks()
    assert (chunk.page, chunk.chunk_index, chunk.source) == (3, 0, "p.pdf")


def test_chunk_ids_are_deterministic():
    assert chunk_ids("doc", 3) == chunk_ids("doc", 3)
    assert chunk_ids("doc", 1) != chunk_ids("other", 1)


def test_legacy_search_shape(store, embed):
    index_chunks("a.md", [Chunk(text="alpha beta")], store, embed)
    out = store.search(embed(["alpha"])[0], top_k=1)
    assert out == {"contexts": ["alpha beta"], "sources": ["a.md"]}
