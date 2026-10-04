import datetime
import logging

import inngest
import inngest.fast_api
from fastapi import FastAPI
from inngest.experimental import ai

from config import settings
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult
from data_loader import load_and_chunk
from embeddings import embed_texts
from pipeline import index_chunks
from prompts import build_answer_messages, build_condense_messages, extract_citations
from retrieval import apply_rerank, build_rerank_messages, retrieve
from vector_db import QdrantStorage

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


def _adapter() -> ai.openai.Adapter:
    return ai.openai.Adapter(auth_key=settings.openai_api_key, model=settings.chat_model)


def _reply_text(res: dict) -> str:
    return (res["choices"][0]["message"]["content"] or "").strip()


@inngest_client.create_function(
    fn_id="RAG: Ingest Document",
    # "rag/ingest_pdf" is kept so older clients keep working
    trigger=[inngest.TriggerEvent(event="rag/ingest_document"), inngest.TriggerEvent(event="rag/ingest_pdf")],
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
    # Collapse rapid duplicate uploads of the same document; the last one wins.
    debounce=inngest.Debounce(period=datetime.timedelta(seconds=10), key="event.data.source_id"),
)
async def rag_ingest_document(ctx: inngest.Context):
    def _load() -> RAGChunkAndSrc:
        path = ctx.event.data.get("path") or ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", path)
        return RAGChunkAndSrc(chunks=load_and_chunk(path), source_id=source_id)

    def _index(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        return index_chunks(chunks_and_src.source_id, chunks_and_src.chunks, QdrantStorage(), embed_texts)

    chunks_and_src = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    result = await ctx.step.run("embed-and-upsert", lambda: _index(chunks_and_src), output_type=RAGUpsertResult)
    return result.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Delete Document",
    trigger=inngest.TriggerEvent(event="rag/delete_document"),
)
async def rag_delete_document(ctx: inngest.Context):
    source_id = ctx.event.data["source_id"]
    removed = await ctx.step.run("delete-chunks", lambda: QdrantStorage().delete_source(source_id))
    return {"source_id": source_id, "removed": removed}


@inngest_client.create_function(
    fn_id="RAG: Query",
    trigger=[inngest.TriggerEvent(event="rag/query"), inngest.TriggerEvent(event="rag/query_pdf_ai")],
)
async def rag_query(ctx: inngest.Context):
    data = ctx.event.data
    question = data["question"].strip()
    top_k = int(data.get("top_k", 5))
    history = data.get("history") or []
    mode = data.get("mode", settings.retrieval_mode)
    rerank = bool(data.get("rerank", settings.rerank))

    # 1. Turn a follow-up ("what about the second one?") into a standalone question
    standalone = question
    if history:
        res = await ctx.step.ai.infer(
            "condense-question",
            adapter=_adapter(),
            body={
                "max_completion_tokens": 200,
                "messages": build_condense_messages(history, question, settings.max_history_turns),
            },
        )
        standalone = _reply_text(res) or question

    # 2. Retrieve (hybrid by default). With re-ranking, fetch a larger pool first.
    pool_k = max(top_k * 3, 10) if rerank else top_k

    def _search() -> RAGSearchResult:
        found = retrieve(standalone, QdrantStorage(), embed_texts, top_k=pool_k, mode=mode,
                         candidate_pool=settings.candidate_pool)
        return RAGSearchResult(contexts=found)

    found = await ctx.step.run("retrieve", _search, output_type=RAGSearchResult)
    contexts = found.contexts

    # 3. Optional LLM re-ranking of the candidates
    if rerank and len(contexts) > top_k:
        res = await ctx.step.ai.infer(
            "rerank",
            adapter=_adapter(),
            body={"max_completion_tokens": 300, "messages": build_rerank_messages(standalone, contexts)},
        )
        contexts = apply_rerank(contexts, _reply_text(res), top_k)
    contexts = contexts[:top_k]

    if not contexts:
        return RAGQueryResult(
            answer="I couldn't find anything relevant in the uploaded documents.",
            standalone_question=standalone, sources=[], citations=[], num_contexts=0, retrieval_mode=mode,
        ).model_dump()

    # 4. Grounded answer with [n] citations
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=_adapter(),
        body={"max_completion_tokens": 1024, "messages": build_answer_messages(standalone, contexts)},
    )
    answer = _reply_text(res)
    return RAGQueryResult(
        answer=answer,
        standalone_question=standalone,
        sources=list(dict.fromkeys(c.source for c in contexts)),
        citations=extract_citations(answer, contexts),
        num_contexts=len(contexts),
        retrieval_mode=mode + ("+rerank" if rerank else ""),
    ).model_dump()


app = FastAPI(title="doc-qa-rag-assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def sources():
    """Documents in the index with their chunk counts (used by the Streamlit sidebar)."""
    return QdrantStorage().list_sources()


inngest.fast_api.serve(app, inngest_client, [rag_ingest_document, rag_delete_document, rag_query])
