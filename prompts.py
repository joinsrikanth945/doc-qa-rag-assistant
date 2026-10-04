"""Prompt construction and citation handling."""
from __future__ import annotations

import re

from custom_types import Citation, RetrievedChunk

ANSWER_SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite every claim with the passage number in square brackets, e.g. [1] or [2][3]. "
    "If the context does not contain the answer, say you could not find it in the documents. "
    "Do not use outside knowledge."
)

CONDENSE_SYSTEM_PROMPT = (
    "Rewrite the user's latest message as a standalone question that can be understood "
    "without the conversation. Keep product names, error codes and numbers exactly. "
    "Reply with the question only."
)


def _label(chunk: RetrievedChunk) -> str:
    return f"{chunk.source}, p. {chunk.page}" if chunk.page else chunk.source


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] ({_label(c)})\n{c.text}" for i, c in enumerate(chunks, start=1))


def build_answer_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    user = (
        f"Context:\n{build_context_block(chunks)}\n\n"
        f"Question: {question}\n"
        "Answer concisely, citing passages like [1]."
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_condense_messages(history: list[dict], question: str, max_turns: int = 6) -> list[dict]:
    recent = history[-max_turns * 2:]
    transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)
    return [
        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"},
    ]


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Return the passages the answer actually cites, in order of first mention."""
    refs = []
    for match in re.findall(r"\[(\d+)\]", answer):
        ref = int(match)
        if 1 <= ref <= len(chunks) and ref not in refs:
            refs.append(ref)
    return [
        Citation(ref=r, source=chunks[r - 1].source, page=chunks[r - 1].page, snippet=chunks[r - 1].text[:300])
        for r in refs
    ]
