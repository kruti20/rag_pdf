from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.models import Chunk, ExtractedSegment

# ~4 characters/token, so ~700 tokens / ~75 token overlap (per tech stack doc)
# maps to these character-based defaults used by RecursiveCharacterTextSplitter.
DEFAULT_CHUNK_SIZE_CHARS = 2800
DEFAULT_CHUNK_OVERLAP_CHARS = 300


def chunk_segments(
    segments: list[ExtractedSegment],
    document_id: str,
    source_type: str,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap_chars
    )

    chunks = []
    for segment in segments:
        for piece in splitter.split_text(segment.text):
            chunks.append(
                Chunk(
                    id=f"{document_id}-{len(chunks)}",
                    document_id=document_id,
                    source_type=source_type,
                    location_label=segment.location_label,
                    text=piece,
                )
            )
    return chunks
