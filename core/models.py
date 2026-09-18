from dataclasses import dataclass, field


@dataclass
class ExtractedSegment:
    text: str
    location_label: str


@dataclass
class Chunk:
    id: str
    document_id: str
    source_type: str  # "pdf" | "docx" | "txt"
    location_label: str
    text: str
    embedding: list[float] | None = None


@dataclass
class Source:
    location_label: str
    snippet: str
    document_id: str = ""


@dataclass
class Answer:
    text: str
    found_in_document: bool
    sources: list[Source] = field(default_factory=list)
