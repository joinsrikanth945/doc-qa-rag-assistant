import asyncio
import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
INNGEST_API_BASE = os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", "uploads"))

st.set_page_config(page_title="Doc Q&A Assistant", page_icon="📄", layout="wide")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def send_event(name: str, data: dict) -> str:
    async def _send():
        ids = await get_inngest_client().send(inngest.Event(name=name, data=data))
        return ids[0]
    return asyncio.run(_send())


def save_upload(file) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOADS_DIR / Path(file.name).name  # strip any directory parts from the name
    path.write_bytes(file.getbuffer())
    return path


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start, last_status = time.time(), None
    while True:
        resp = requests.get(f"{INNGEST_API_BASE}/events/{event_id}/runs", timeout=10)
        resp.raise_for_status()
        runs = resp.json().get("data", [])
        if runs:
            status = runs[0].get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return runs[0].get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


def fetch_sources() -> dict[str, int]:
    try:
        resp = requests.get(f"{API_BASE}/sources", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {}


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            where = f"{c['source']}, page {c['page']}" if c.get("page") else c["source"]
            st.markdown(f"**[{c['ref']}] {where}**")
            st.caption(c["snippet"] + ("…" if len(c["snippet"]) >= 300 else ""))


# ---------------- Sidebar: documents and settings ----------------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add documents", type=SUPPORTED_TYPES, accept_multiple_files=True)
    if uploads and st.button(f"Ingest {len(uploads)} file(s)", type="primary"):
        for file in uploads:
            path = save_upload(file)
            send_event("rag/ingest_document", {"path": str(path.resolve()), "source_id": path.name})
        st.success("Ingestion started. Documents appear below once processed.")

    sources = fetch_sources()
    if sources:
        for name, n_chunks in sources.items():
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {name}  \n<small>{n_chunks} chunks</small>", unsafe_allow_html=True)
            if col2.button("🗑", key=f"del-{name}", help=f"Remove {name} from the index"):
                send_event("rag/delete_document", {"source_id": name})
                st.toast(f"Removing {name}…")
        if st.button("Refresh list"):
            st.rerun()
    else:
        st.caption("No documents indexed yet (or the API is not reachable).")

    st.header("Settings")
    mode = st.radio("Retrieval", ["hybrid", "dense"], horizontal=True,
                    help="Hybrid combines semantic search with keyword (BM25) search.")
    rerank = st.toggle("LLM re-ranking", value=False, help="Slower, but often picks better passages.")
    top_k = st.slider("Passages to use", 1, 15, 5)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------- Chat ----------------
st.title("Ask your documents")
st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))

if question := st.chat_input("Ask a question about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and generating an answer…"):
                event_id = send_event("rag/query", {
                    "question": question, "top_k": top_k, "history": history,
                    "mode": mode, "rerank": rerank,
                })
                output = wait_for_run_output(event_id)
            answer = output.get("answer") or "(No answer)"
            citations = output.get("citations", [])
            st.markdown(answer)
            render_citations(citations)
            if output.get("standalone_question") and output["standalone_question"] != question:
                st.caption(f"Searched for: _{output['standalone_question']}_")
        except Exception as exc:  # show the problem instead of crashing the app
            answer, citations = f"Something went wrong: {exc}", []
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "citations": citations})
