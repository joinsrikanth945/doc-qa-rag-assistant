import pytest

from data_loader import UnsupportedFileType, chunk_pages, load_and_chunk, load_pages


def test_markdown_is_loaded_and_chunked(sample_docs):
    chunks = load_and_chunk(sample_docs / "thermostat_manual.md", chunk_size=128, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)
    assert all(c.page is None for c in chunks)  # markdown has no pages
    assert any("E47" in c.text for c in chunks)


def test_page_numbers_are_kept():
    pages = [(1, "First page text. " * 50), (2, "Second page text. " * 50)]
    chunks = chunk_pages(pages, chunk_size=64, chunk_overlap=0)
    assert {c.page for c in chunks} == {1, 2}
    assert all(("First" in c.text) == (c.page == 1) for c in chunks)


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([(1, "   "), (2, "")]) == []


def test_unsupported_extension(tmp_path):
    f = tmp_path / "data.xlsx"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFileType):
        load_pages(f)


def test_docx_is_supported(tmp_path):
    docx = pytest.importorskip("docx")  # python-docx, only used to build a fixture
    path = tmp_path / "note.docx"
    d = docx.Document()
    d.add_paragraph("The reset button is on the back of the device.")
    d.save(path)
    chunks = load_and_chunk(path)
    assert "reset button" in chunks[0].text
