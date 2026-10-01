# ProductionGradeRAGPythonApp

A production-leaning Retrieval-Augmented Generation (RAG) application in Python: ingest PDF documents, ask questions about them in natural language, and get grounded answers with source citations.

This started as a reference implementation of a RAG pattern I also use internally to make Temenos WealthSuite product documentation searchable in natural language, cutting down time spent digging through manuals and internal wikis.

## Features

- PDF ingestion and chunking pipeline (LlamaIndex)
- Vector storage and semantic search via Qdrant (local instance, cosine similarity)
- Event-driven, durable workflows via Inngest — ingestion and querying are modeled as Inngest functions with automatic retries
- Built-in throttling and per-source rate limiting on ingestion, so repeated or bulk uploads don't overwhelm the pipeline
- OpenAI LLM for grounded answer generation, with responses returned alongside the source documents they were drawn from
- FastAPI backend exposing ingestion and query as HTTP endpoints
- Streamlit front-end for interactive querying

## Tech stack

- Python 3.13
- FastAPI + Uvicorn
- Inngest for event-driven workflow orchestration
- LlamaIndex (`llama-index-core`, `llama-index-readers-file`) for document loading and chunking
- Qdrant (`qdrant-client`) as the vector database
- OpenAI API for embeddings and answer generation
- Streamlit for the UI
- `uv` for dependency management

## How it works

1. **Ingest** — a PDF is loaded and split into chunks, each chunk is embedded and upserted into Qdrant with an ID derived from the source document and chunk index.
2. **Query** — a user's question is embedded and used to search Qdrant for the most relevant chunks; those chunks are passed as context to an OpenAI model, which generates an answer along with the source documents it drew from.
3. Both steps run as Inngest functions, which gives them retries, throttling and rate limiting out of the box, and are exposed over HTTP via FastAPI.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | FastAPI app and Inngest functions (`rag_ingest_pdf`, `rag_query_pdf_ai`) |
| `vector_db.py` | `QdrantStorage` class — upsert and similarity search against Qdrant |
| `data_loader.py` | PDF loading and chunking logic |
| `ingest_all.py` | Batch ingestion entry point |
| `custom_types.py` | Shared type/schema definitions |
| `streamlit_app.py` | Streamlit UI for querying |
| `qdrant_local_db/`, `qdrant_storage/` | Local Qdrant persistence |

## Getting started

### Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) package manager
- An OpenAI API key
- A running Inngest dev server for local event handling

### Installation

```bash
git clone https://github.com/joinsrikanth945/ProductionGradeRAGPythonApp.git
cd ProductionGradeRAGPythonApp
uv sync
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

### Run

```bash
# start the FastAPI / Inngest backend
uv run uvicorn main:app --reload

# in a separate terminal, start the Streamlit UI
uv run streamlit run streamlit_app.py
```

*(Double-check the exact run commands and entry-point names against `main.py` — adjust if your FastAPI app object or entry point is named differently.)*

## License

Add a license (e.g. MIT) here if you intend to keep this public.﻿# ProductionGradeRAGPythonApp

Results obtained 

<img width="1379" height="916" alt="image" src="https://github.com/user-attachments/assets/afe62d13-a460-4f89-bb58-a6aabb49be81" />




