from pathlib import Path

import docx
import pymupdf
import pytest

from core.ingest import FileTooLargeError, UnsupportedFileTypeError, ingest_document

FIXTURES = Path(__file__).parent / "fixtures"


def _build_docx(path: Path, blocks: list[tuple[str, str]]):
    """blocks: list of (style, text) pairs, e.g. ("Heading 1", "Termination")."""
    document = docx.Document()
    for style, text in blocks:
        document.add_paragraph(text, style=style)
    document.save(str(path))


def _build_pdf(path: Path, pages: list[str]):
    """pages: list of text per page; empty string means a blank page."""
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(str(path))
    pdf.close()


def test_rejects_legacy_doc_extension():
    with pytest.raises(UnsupportedFileTypeError, match=r"\.docx"):
        ingest_document("contract.doc")


def test_rejects_unknown_extension():
    with pytest.raises(UnsupportedFileTypeError):
        ingest_document("image.png")


def test_loads_txt_with_line_range_labels():
    result = ingest_document(str(FIXTURES / "sample.txt"))

    assert result.source_type == "txt"
    assert result.has_extractable_text is True
    assert [seg.location_label for seg in result.segments] == [
        "lines 1-20",
        "lines 21-40",
        "lines 41-45",
    ]
    assert "Line 01" in result.segments[0].text
    assert "Line 20" in result.segments[0].text
    assert "Line 21" in result.segments[1].text
    assert "Line 45" in result.segments[2].text


def test_loads_docx_with_paragraph_and_nearest_heading_labels(tmp_path):
    docx_path = tmp_path / "contract.docx"
    _build_docx(
        docx_path,
        [
            ("Heading 1", "Introduction"),
            ("Normal", "This is the introduction paragraph text."),
            ("Heading 1", "Termination"),
            ("Normal", "Either party may terminate with 30 days written notice."),
            ("Normal", "Termination fees apply if notice is not given."),
        ],
    )

    result = ingest_document(str(docx_path))

    assert result.source_type == "docx"
    assert result.has_extractable_text is True
    labels = [seg.location_label for seg in result.segments]
    assert labels == [
        "para 1 (Introduction)",
        "para 2 (Introduction)",
        "para 3 (Termination)",
        "para 4 (Termination)",
        "para 5 (Termination)",
    ]
    assert result.segments[3].text == "Either party may terminate with 30 days written notice."


def test_docx_paragraph_before_any_heading_uses_document_fallback(tmp_path):
    docx_path = tmp_path / "no_heading.docx"
    _build_docx(docx_path, [("Normal", "No heading has appeared yet.")])

    result = ingest_document(str(docx_path))

    assert result.segments[0].location_label == "para 1 (Document)"


def test_docx_skips_blank_paragraphs(tmp_path):
    docx_path = tmp_path / "blank.docx"
    _build_docx(
        docx_path,
        [
            ("Normal", "First paragraph."),
            ("Normal", "   "),
            ("Normal", "Second paragraph."),
        ],
    )

    result = ingest_document(str(docx_path))

    assert [seg.text for seg in result.segments] == ["First paragraph.", "Second paragraph."]


def test_loads_pdf_with_page_labels(tmp_path):
    pdf_path = tmp_path / "contract.pdf"
    _build_pdf(pdf_path, ["Page one content.", "Page two content."])

    result = ingest_document(str(pdf_path))

    assert result.source_type == "pdf"
    assert result.has_extractable_text is True
    assert [seg.location_label for seg in result.segments] == ["p.1", "p.2"]
    assert "Page one content." in result.segments[0].text
    assert "Page two content." in result.segments[1].text


def test_scanned_pdf_with_no_extractable_text_does_not_crash(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    _build_pdf(pdf_path, ["", ""])

    result = ingest_document(str(pdf_path))

    assert result.source_type == "pdf"
    assert result.has_extractable_text is False
    assert result.segments == []


def test_rejects_file_larger_than_20mb(tmp_path):
    large_path = tmp_path / "large.txt"
    with open(large_path, "wb") as f:
        f.seek(21 * 1024 * 1024 - 1)
        f.write(b"\0")

    with pytest.raises(FileTooLargeError, match=r"20\s*MB"):
        ingest_document(str(large_path))


def test_accepts_file_at_the_20mb_boundary(tmp_path):
    at_limit_path = tmp_path / "at_limit.txt"
    with open(at_limit_path, "wb") as f:
        f.seek(20 * 1024 * 1024 - 1)
        f.write(b"\0")

    result = ingest_document(str(at_limit_path))  # must not raise

    assert result.source_type == "txt"
