# doc-qa-rag-assistant

A production-leaning Retrieval-Augmented Generation (RAG) application in Python: ingest PDF documents, ask questions about them in natural language, and get grounded answers with source citations.

I adapted this pattern to make long technical product documentation searchable in natural language, cutting down time spent digging through manuals and internal wikis.

## Credits

This project is based on Tech With Tim's [Production-Grade RAG Python App](https://github.com/techwithtim/ProductionGradeRAGPythonApp) tutorial. My changes:

- Batch ingestion of a folder of PDFs (`ingest_all.py`)
- Local file-based Qdrant storage instead of a Qdrant server
- Updated OpenAI model and request parameters
- Rewritten documentation and result screenshots

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

## Getting started

### Prerequisites

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) package manager
- An OpenAI API key
- A running Inngest dev server for local event handling

### Installation

```bash
git clone https://github.com/joinsrikanth945/doc-qa-rag-assistant.git
cd doc-qa-rag-assistant
uv sync
```

### Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

### Run

```bash
# terminal 1: start the FastAPI / Inngest backend
uv run uvicorn main:app --reload --port 8000

# terminal 2: start the Inngest dev server
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest

# terminal 3: start the Streamlit UI
uv run streamlit run streamlit_app.py
```

Then open http://localhost:8501 for the app and http://localhost:8288 for the Inngest dashboard.

## Results

![Answer with sources in the Streamlit app](docs/images/answer.png)

<img width="1831" height="873" alt="image" src="https://github.com/user-attachments/assets/f2cf3d1c-23c5-4403-aff8-2a82c9fecc4b" />

<img width="1294" height="640" alt="image" src="https://github.com/user-attachments/assets/d58d99ce-a374-467f-a1f7-9e1e97977676" />

