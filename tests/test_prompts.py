from custom_types import RetrievedChunk
from prompts import build_answer_messages, build_condense_messages, build_context_block, extract_citations

CHUNKS = [
    RetrievedChunk(text="E12 means the C-wire lost power.", source="manual.pdf", page=7),
    RetrievedChunk(text="Hub uses port 8883.", source="guide.md"),
]


def test_context_block_numbers_and_labels():
    block = build_context_block(CHUNKS)
    assert block.startswith("[1] (manual.pdf, p. 7)")
    assert "[2] (guide.md)" in block


def test_answer_messages_contain_question_and_rules():
    system, user = build_answer_messages("What is E12?", CHUNKS)
    assert "Cite" in system["content"]
    assert "What is E12?" in user["content"] and "[2]" in user["content"]


def test_extract_citations_dedupes_and_ignores_out_of_range():
    cites = extract_citations("It lost power [1]. Use port 8883 [2][1]. See [5].", CHUNKS)
    assert [(c.ref, c.source, c.page) for c in cites] == [(1, "manual.pdf", 7), (2, "guide.md", None)]


def test_condense_uses_only_recent_history():
    history = [{"role": "user", "content": f"q{i}"} for i in range(20)]
    _, user = build_condense_messages(history, "and the next one?", max_turns=2)
    assert "q19" in user["content"] and "q10" not in user["content"]
