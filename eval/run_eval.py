"""Evaluate retrieval (and optionally answers) on a labelled question set.

Usage (from the repo root):
    python eval/run_eval.py                 # OpenAI embeddings, compares dense vs hybrid
    python eval/run_eval.py --offline       # no API key needed (hashing embedder)
    python eval/run_eval.py --answers       # also generate answers and check them

Metrics
    hit@k  share of questions where a retrieved chunk contains the expected evidence
    MRR    mean reciprocal rank of the first chunk containing the evidence
    answer accuracy (with --answers)  share of answers containing all expected keywords
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import SUPPORTED_EXTENSIONS, settings  # noqa: E402
from data_loader import load_and_chunk  # noqa: E402
from embeddings import HashEmbedder, embed_texts  # noqa: E402
from pipeline import index_chunks  # noqa: E402
from prompts import build_answer_messages  # noqa: E402
from retrieval import retrieve  # noqa: E402
from vector_db import QdrantStorage  # noqa: E402


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def first_hit_rank(chunks, case) -> int | None:
    evidence = _normalize(case["evidence"])
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.source == case["source"] and evidence in _normalize(chunk.text):
            return rank
    return None


def build_store(docs_dir: Path, embed_fn, dim: int, chunk_size: int, overlap: int) -> QdrantStorage:
    store = QdrantStorage(location=":memory:", collection=f"eval_{int(time.time() * 1000)}", dim=dim)
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    for path in files:
        chunks = load_and_chunk(path, chunk_size=chunk_size, chunk_overlap=overlap)
        index_chunks(path.name, chunks, store, embed_fn)
    return store


def evaluate(cases, store, embed_fn, mode: str, k: int) -> dict:
    hits, rr, per_case = 0, 0.0, []
    for case in cases:
        found = retrieve(case["question"], store, embed_fn, top_k=k, mode=mode)
        rank = first_hit_rank(found, case)
        hits += rank is not None
        rr += 1.0 / rank if rank else 0.0
        per_case.append({"question": case["question"], "rank": rank, "chunks": found})
    n = len(cases)
    return {"mode": mode, "hit_at_k": hits / n, "mrr": rr / n, "per_case": per_case}


def check_answers(cases, result) -> float:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    correct = 0
    for case, row in zip(cases, result["per_case"]):
        reply = client.chat.completions.create(
            model=settings.chat_model,
            messages=build_answer_messages(case["question"], row["chunks"]),
        ).choices[0].message.content or ""
        ok = all(kw.lower() in reply.lower() for kw in case["answer_keywords"])
        correct += ok
        print(f"  {'✔' if ok else '✘'} {case['question']}")
    return correct / len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=ROOT / "eval" / "sample_docs")
    parser.add_argument("--set", dest="eval_set", type=Path, default=ROOT / "eval" / "eval_set.jsonl")
    parser.add_argument("-k", type=int, default=3, help="chunks retrieved per question")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--offline", action="store_true", help="use the hashing embedder (no API key)")
    parser.add_argument("--answers", action="store_true", help="also generate and check answers (needs a key)")
    parser.add_argument("--verbose", action="store_true", help="list the rank for every question")
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    if args.offline:
        embed_fn, dim = HashEmbedder(dim=256), 256
    else:
        embed_fn, dim = embed_texts, settings.embed_dim

    store = build_store(args.docs, embed_fn, dim, args.chunk_size, args.chunk_overlap)
    print(f"{len(cases)} questions · {sum(store.list_sources().values())} chunks · k={args.k} · "
          f"embedder={'hashing (offline)' if args.offline else settings.embed_model}\n")

    results = [evaluate(cases, store, embed_fn, mode, args.k) for mode in ("dense", "hybrid")]

    print(f"| Retrieval | hit@{args.k} | MRR |")
    print("|---|---|---|")
    for r in results:
        print(f"| {r['mode']} | {r['hit_at_k']:.0%} | {r['mrr']:.2f} |")

    if args.verbose:
        for r in results:
            print(f"\n{r['mode']}:")
            for row in r["per_case"]:
                print(f"  rank={row['rank'] or '-':>2}  {row['question']}")

    if args.answers:
        if args.offline or not settings.openai_api_key:
            print("\n--answers needs OPENAI_API_KEY and cannot be combined with --offline")
            return 1
        print("\nAnswer check (hybrid retrieval):")
        accuracy = check_answers(cases, results[1])
        print(f"Answer accuracy: {accuracy:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
