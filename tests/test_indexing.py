from pathlib import Path

from core.embedder import embed_texts
from core.indexing import index_document
from core.vectorstore import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"


def test_index_document_wires_ingest_chunk_embed_store(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    result = index_document(
        str(FIXTURES / "sample.txt"), document_id="txt-doc", vector_store=store
    )

    assert result.has_extractable_text is True
    query_embedding = embed_texts(["The quick brown fox jumps over the lazy dog."])[0]
    matches = store.query("txt-doc", query_embedding=query_embedding, k=1)
    assert len(matches) == 1
    assert matches[0]["location_label"].startswith("lines ")


def test_index_document_reports_chunk_count(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    result = index_document(
        str(FIXTURES / "sample.txt"), document_id="txt-doc", vector_store=store
    )

    # sample.txt has 45 lines -> 3 segments of <=20 lines, none long enough to split further
    assert result.chunk_count == 3


def test_index_document_scanned_pdf_reports_zero_chunks(tmp_path):
    import pymupdf

    scanned_path = tmp_path / "scanned.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(str(scanned_path))
    pdf.close()

    store = VectorStore(persist_directory=str(tmp_path))
    result = index_document(str(scanned_path), document_id="scanned-doc", vector_store=store)

    assert result.has_extractable_text is False
    assert result.chunk_count == 0


def test_query_known_phrase_from_real_pdf_returns_correct_page(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf"),
        document_id="pdf-doc",
        vector_store=store,
    )

    query_embedding = embed_texts(["What are prompt injection attacks?"])[0]
    matches = store.query("pdf-doc", query_embedding=query_embedding, k=3)

    assert any(m["location_label"] == "p.22" for m in matches)
