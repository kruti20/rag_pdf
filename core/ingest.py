import os

import docx
import pymupdf

from core.models import ExtractedSegment

TXT_LINES_PER_SEGMENT = 20
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20MB, per PRD


class DocumentValidationError(Exception):
    pass


class UnsupportedFileTypeError(DocumentValidationError):
    pass


class FileTooLargeError(DocumentValidationError):
    pass


class IngestResult:
    def __init__(self, segments: list[ExtractedSegment], source_type: str):
        self.segments = segments
        self.source_type = source_type
        self.has_extractable_text = any(seg.text.strip() for seg in segments)
        self.chunk_count = 0


def _load_txt(file_path: str) -> IngestResult:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    segments = []
    for start in range(0, len(lines), TXT_LINES_PER_SEGMENT):
        block = lines[start : start + TXT_LINES_PER_SEGMENT]
        first_line = start + 1
        last_line = start + len(block)
        segments.append(
            ExtractedSegment(
                text="\n".join(block),
                location_label=f"lines {first_line}-{last_line}",
            )
        )
    return IngestResult(segments=segments, source_type="txt")


def _load_docx(file_path: str) -> IngestResult:
    document = docx.Document(file_path)

    segments = []
    current_heading = "Document"
    index = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        index += 1
        if paragraph.style.name.startswith("Heading"):
            current_heading = text
        segments.append(
            ExtractedSegment(
                text=text,
                location_label=f"para {index} ({current_heading})",
            )
        )
    return IngestResult(segments=segments, source_type="docx")


def _load_pdf(file_path: str) -> IngestResult:
    segments = []
    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if not text:
                continue
            segments.append(
                ExtractedSegment(text=text, location_label=f"p.{page_number}")
            )
    return IngestResult(segments=segments, source_type="pdf")


def ingest_document(file_path: str) -> IngestResult:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".doc":
        raise UnsupportedFileTypeError(
            "Old Word format (.doc) isn't supported — please save as .docx and re-upload."
        )
    if ext not in (".pdf", ".docx", ".txt"):
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext or '(no extension)'}")

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File is {size / (1024 * 1024):.1f}MB, which exceeds the 20MB limit — "
            "please upload a smaller file."
        )

    if ext == ".txt":
        return _load_txt(file_path)
    if ext == ".docx":
        return _load_docx(file_path)
    return _load_pdf(file_path)
