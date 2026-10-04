from custom_types import Chunk, RetrievedChunk
from pipeline import index_chunks
from retrieval import apply_rerank, bm25_search, reciprocal_rank_fusion, retrieve, tokenize


def rc(text, source="doc", i=0):
    return RetrievedChunk(text=text, source=source, chunk_index=i)


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("What does error E-47 mean on the N4?") == ["error", "e-47", "mean", "n4"]


def test_bm25_prefers_exact_term():
    chunks = [rc("Restart the device to fix most issues", i=0), rc("Error E31 means a sensor fault", i=1)]
    top = bm25_search("what is E31", chunks, top_k=2)
    assert top[0].chunk_index == 1


def test_bm25_without_matches_returns_nothing():
    assert bm25_search("zebra", [rc("nothing relevant here")], top_k=3) == []


def test_rrf_rewards_agreement():
    a, b, c = rc("a", i=0), rc("b", i=1), rc("c", i=2)
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [x.chunk_index for x in fused] == [1, 0, 2]  # b is near the top of both lists
    assert fused[0].score > fused[-1].score


def test_rerank_reorders_and_falls_back():
    chunks = [rc("x", i=0), rc("y", i=1), rc("z", i=2)]
    assert [c.chunk_index for c in apply_rerank(chunks, '{"ranking": [2, 0]}', 3)] == [2, 0, 1]
    assert [c.chunk_index for c in apply_rerank(chunks, "not json", 2)] == [0, 1]
    assert [c.chunk_index for c in apply_rerank(chunks, '{"ranking": [9, "a", 1, 1]}', 3)] == [1, 0, 2]


def test_hybrid_and_dense_end_to_end(store, embed):
    index_chunks("manual.md", [
        Chunk(text="Error E12 means the C-wire lost power.", page=None),
        Chunk(text="The warranty lasts three years from purchase.", page=None),
    ], store, embed)
    index_chunks("guide.pdf", [Chunk(text="Pairing mode lasts 120 seconds.", page=4)], store, embed)

    for mode in ("dense", "hybrid"):
        found = retrieve("What does E12 mean?", store, embed, top_k=2, mode=mode)
        assert found[0].source == "manual.md" and "E12" in found[0].text

    found = retrieve("how long is pairing mode", store, embed, top_k=1)
    assert found[0].page == 4
