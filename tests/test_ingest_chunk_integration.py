import re
from pathlib import Path

from core.chunker import chunk_segments
from core.ingest import ingest_document

FIXTURES = Path(__file__).parent / "fixtures"


def test_real_pdf_produces_well_formed_chunks():
    result = ingest_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf")
    )
    assert result.has_extractable_text is True

    chunks = chunk_segments(result.segments, document_id="pdf-doc", source_type="pdf")

    assert len(chunks) > 0
    assert all(re.fullmatch(r"p\.\d+", c.location_label) for c in chunks)
    assert all(c.text.strip() for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)  # ids unique
    # a 24-page real document should span multiple distinct pages
    assert len({c.location_label for c in chunks}) > 1


def test_real_docx_produces_well_formed_chunks():
    result = ingest_document(
        str(FIXTURES / "Build a RAG Application with LlamaIndex.docx")
    )
    assert result.has_extractable_text is True

    chunks = chunk_segments(result.segments, document_id="docx-doc", source_type="docx")

    assert len(chunks) > 0
    assert all(re.fullmatch(r"para \d+ \(.+\)", c.location_label) for c in chunks)
    assert all(c.text.strip() for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_sample_txt_produces_well_formed_chunks():
    result = ingest_document(str(FIXTURES / "sample.txt"))
    assert result.has_extractable_text is True

    chunks = chunk_segments(result.segments, document_id="txt-doc", source_type="txt")

    assert len(chunks) == 3  # 45 lines / 20 lines-per-segment, no oversized segment to split
    assert all(re.fullmatch(r"lines \d+-\d+", c.location_label) for c in chunks)
