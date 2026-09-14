from dataclasses import replace

from core.chunker import chunk_segments
from core.embedder import embed_texts
from core.ingest import IngestResult, ingest_document
from core.vectorstore import VectorStore


def index_document(
    file_path: str, document_id: str, vector_store: VectorStore
) -> IngestResult:
    result = ingest_document(file_path)
    if not result.has_extractable_text:
        return result

    chunks = chunk_segments(result.segments, document_id=document_id, source_type=result.source_type)
    embeddings = embed_texts([c.text for c in chunks])
    embedded_chunks = [replace(c, embedding=e) for c, e in zip(chunks, embeddings)]

    vector_store.add_chunks(embedded_chunks)
    result.chunk_count = len(chunks)
    return result
