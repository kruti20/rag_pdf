import os

import docx
import docx.opc.exceptions
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


class CorruptDocumentError(DocumentValidationError):
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
    try:
        document = docx.Document(file_path)
    except docx.opc.exceptions.PackageNotFoundError:
        raise CorruptDocumentError(
            "This Word file appears to be corrupted or is not a valid .docx file — "
            "please try re-saving it from Word and re-uploading."
        )
    except Exception as exc:
        raise CorruptDocumentError(
            f"Could not open the Word file ({exc}) — it may be corrupted or password-protected."
        ) from exc

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
    try:
        pdf = pymupdf.open(file_path)
    except pymupdf.FileDataError as exc:
        raise CorruptDocumentError(
            "This PDF file appears to be corrupted and could not be opened — "
            "please check the file and re-upload."
        ) from exc

    with pdf:
        # Encrypted PDFs open without error but require a password to read pages.
        if pdf.needs_pass:
            raise CorruptDocumentError(
                "This PDF is password-protected — please remove the password and re-upload."
            )

        segments = []
        for page_number, page in enumerate(pdf, start=1):
            try:
                text = page.get_text().strip()
            except (ValueError, RuntimeError) as exc:
                raise CorruptDocumentError(
                    f"Could not read page {page_number} of the PDF — "
                    "the file may be corrupted or encrypted."
                ) from exc
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
