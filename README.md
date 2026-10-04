# doc-qa-rag-assistant

Conversational question answering over your own documents (PDF, Word, Markdown, text). Answers are grounded in the documents and cite the exact passage and page they came from.

I built this to make long technical product documentation searchable in natural language, cutting down time spent digging through manuals and internal wikis.

## Origin and credit

The project started from a public tutorial on building a production-style RAG app with FastAPI, Inngest, Qdrant and Streamlit: [TUTORIAL NAME](TUTORIAL-LINK). The tutorial supplied the basic ingest → embed → search → answer pipeline and the Inngest/Streamlit wiring. Everything listed under **What I added** below is my own work on top of it.

## What I added

| Area | Change | Why |
|---|---|---|
| Retrieval | **Hybrid search**: dense vectors + BM25 keyword search, merged with Reciprocal Rank Fusion (`retrieval.py`) | Embeddings blur exact terms like error codes and model numbers; keyword search catches them |
| Retrieval | Optional **LLM re-ranking** of a larger candidate pool | Puts the most useful passages first when the top hits are close |
| Answers | **Numbered citations with page numbers**; the UI shows which passages were cited | Users can verify every claim against the source |
| Answers | Stricter grounding prompt: says when the answer is not in the documents | Fewer made-up answers |
| Chat | **Multi-turn chat** with follow-up questions rewritten into standalone queries | "What about the second one?" works |
| Ingestion | **PDF, DOCX, TXT and MD** support, page-aware chunking, batched embeddings | Real documentation isn't only PDFs; large files no longer hit request limits |
| Ingestion | **Re-ingesting replaces** a document's old chunks; documents can be **deleted** | The original kept stale chunks and blocked re-uploads for 4 hours |
| Fixes | Replaced `QdrantClient.search`, which current `qdrant-client` versions removed, with `query_points` | The original code fails on current library versions |
| Quality | **Evaluation script** with a labelled question set, measuring hit@k and MRR (`eval/`) | Retrieval changes are measured, not guessed |
| Quality | **20 unit tests** (pytest), **GitHub Actions CI** | Safe refactoring |
| Ops | **Docker Compose** stack (API, Inngest, Qdrant, UI), config via environment variables (`config.py`) | One command to run everything |

## Results

Retrieval on the included evaluation set (15 questions over two fictional product manuals, top 3 chunks):

| Retrieval | hit@3 | MRR |
|---|---|---|
| dense only | 87% | 0.73 |
| **hybrid (dense + BM25)** | **100%** | **0.87** |

These numbers come from the offline hashing embedder used in CI (`python eval/run_eval.py --offline`). To measure with OpenAI embeddings and check generated answers, run `python eval/run_eval.py --answers` with your API key set.

![Answer with sources in the Streamlit app](docs/images/answer.png)

<img width="1831" height="873" alt="image" src="https://github.com/user-attachments/assets/f2cf3d1c-23c5-4403-aff8-2a82c9fecc4b" />

<img width="1294" height="640" alt="image" src="https://github.com/user-attachments/assets/d58d99ce-a374-467f-a1f7-9e1e97977676" />

## Architecture

```
Streamlit UI ──event──▶ Inngest ──▶ FastAPI (Inngest functions)
                                      ├─ RAG: Ingest Document  load → chunk (with pages) → embed → replace in Qdrant
                                      ├─ RAG: Delete Document
                                      └─ RAG: Query            condense follow-up → hybrid retrieve
                                                               → (re-rank) → answer with [n] citations
```

Ingestion and querying run as Inngest functions, which provides retries, throttling, debouncing and a dashboard showing every step.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app and Inngest functions (ingest, delete, query); `/health` and `/sources` endpoints |
| `retrieval.py` | BM25, Reciprocal Rank Fusion, hybrid retrieval, re-ranking |
| `prompts.py` | Answer and follow-up prompts, citation extraction |
| `data_loader.py` | Multi-format loading and page-aware chunking |
| `embeddings.py` | Batched OpenAI embeddings, plus an offline hashing embedder for tests |
| `pipeline.py` | Indexing logic shared by the app and the evaluation |
| `vector_db.py` | Qdrant wrapper (local or server) |
| `config.py` | All settings, from environment variables |
| `streamlit_app.py` | Chat UI with document management |
| `ingest_all.py` | Batch-ingest a folder |
| `eval/` | Evaluation script, question set and sample documents |
| `tests/` | Unit and integration tests |

## Getting started

### With Docker

```bash
cp .env.example .env   # add your OPENAI_API_KEY
docker compose up --build
```

Open http://localhost:8501 for the app and http://localhost:8288 for the Inngest dashboard.

### Without Docker

Prerequisites: Python 3.13+, [`uv`](https://docs.astral.sh/uv/), Node.js (for the Inngest dev server), an OpenAI API key.

```bash
git clone https://github.com/joinsrikanth945/doc-qa-rag-assistant.git
cd doc-qa-rag-assistant
uv sync
cp .env.example .env   # add your OPENAI_API_KEY

# terminal 1: API
uv run uvicorn main:app --reload --port 8000
# terminal 2: Inngest dev server
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest
# terminal 3: UI
uv run streamlit run streamlit_app.py
```

Batch-ingest a folder: `uv run python ingest_all.py ./my_docs`

### Tests and evaluation

```bash
uv run pytest -q
uv run python eval/run_eval.py --offline --verbose
```

## Configuration

All settings live in `.env` (see `.env.example`): models, chunk size, retrieval mode (`hybrid`/`dense`), re-ranking, and the Qdrant location. Set `QDRANT_URL` to use a Qdrant server; otherwise a local on-disk database is used.

## Possible next steps

- Store BM25 as Qdrant sparse vectors so keyword search scales beyond small collections (it currently scores all chunks in memory)
- Stream answers token by token
- Authentication and per-user document collections
