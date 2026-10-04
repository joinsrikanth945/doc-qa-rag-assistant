"""Load documents of several formats and split them into page-aware chunks."""
from __future__ import annotations

from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from config import SUPPORTED_EXTENSIONS, settings
from custom_types import Chunk
from embeddings import embed_texts  # noqa: F401  (re-exported for backwards compatibility)


class UnsupportedFileType(ValueError):
    pass


def _splitter(chunk_size: int | None = None, chunk_overlap: int | None = None) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
    )


def load_pages(path: str | Path) -> list[tuple[int | None, str]]:
    """Return (page_number, text) pairs. Formats without pages return a single (None, text)."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        from llama_index.readers.file import PDFReader

        docs = PDFReader().load_data(file=path)
        pages = []
        for i, doc in enumerate(docs, start=1):
            text = getattr(doc, "text", "") or ""
            label = (doc.metadata or {}).get("page_label")
            page = int(label) if str(label).isdigit() else i
            if text.strip():
                pages.append((page, text))
        return pages

    if ext == ".docx":
        import docx2txt

        return [(None, docx2txt.process(str(path)) or "")]

    if ext in (".txt", ".md"):
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]

    raise UnsupportedFileType(f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")


def chunk_pages(
    pages: list[tuple[int | None, str]], chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[Chunk]:
    splitter = _splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for page, text in pages:
        for piece in splitter.split_text(text):
            if piece.strip():
                chunks.append(Chunk(text=piece.strip(), page=page))
    return chunks


def load_and_chunk(path: str | Path, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]:
    return chunk_pages(load_pages(path), chunk_size, chunk_overlap)


# Kept for backwards compatibility with the original API
def load_and_chunk_pdf(path: str) -> list[str]:
    return [c.text for c in load_and_chunk(path)]
