import re

from core.embedder import embed_texts
from core.vectorstore import VectorStore

FACTUAL_K = 5
BROAD_K = 12
SUMMARY_MAX_CHUNKS_PER_DOCUMENT = 60
# Chunk count alone doesn't bound prompt size when chunks are near the max
# chunk size — a 305-chunk production document hit Groq's free-tier 8000
# tokens-per-minute limit with a 60-chunk (~9000-token) summarization
# prompt. ~4 chars/token (this project's existing heuristic, see
# core/chunker.py) puts 20000 chars at ~5000 tokens, safely under that limit
# with headroom for the system prompt and multiple loaded documents.
SUMMARY_MAX_CONTEXT_CHARS = 20000

_EXHAUSTIVE_PATTERNS = [
    r"\bfind all\b",
    r"\ball references\b",
    r"\bevery reference\b",
    r"\bevery mention\b",
    r"\ball instances\b",
    r"\bevery instance\b",
    r"\blist all\b",
    r"\ball obligations\b",
]
_SUMMARIZATION_PATTERNS = [
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\bsynthesize\b",
]


def detect_question_type(question: str) -> str:
    lowered = question.lower()
    if any(re.search(p, lowered) for p in _EXHAUSTIVE_PATTERNS):
        return "exhaustive"
    if any(re.search(p, lowered) for p in _SUMMARIZATION_PATTERNS):
        return "summarization"
    return "factual"


_TRIGGER_PHRASES = [
    "find all references to",
    "find all instances of",
    "list all",
    "find all",
    "all references to",
    "all instances of",
    "every reference to",
    "every mention of",
    "every instance of",
    "give me all",
]
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "in", "on", "of", "to",
    "is", "are", "was", "were", "does", "do", "did", "can", "could", "please",
    "you", "me", "give", "what", "who", "when", "where", "why", "how", "i",
    "contract", "document", "about", "for",
}


def extract_keywords(question: str) -> list[str]:
    lowered = question.lower().rstrip("?").strip()
    for phrase in _TRIGGER_PHRASES:
        lowered = lowered.replace(phrase, "")
    words = re.findall(r"[a-z0-9]+", lowered)
    return [w for w in words if w not in _STOPWORDS]


def keyword_search(chunks: list[dict], keywords: list[str]) -> list[dict]:
    if not keywords:
        return []
    return [
        chunk
        for chunk in chunks
        if any(keyword.lower() in chunk["text"].lower() for keyword in keywords)
    ]


def dedupe_by_id(chunks: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for chunk in chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            result.append(chunk)
    return result


def _evenly_spaced_sample(chunks: list[dict], max_count: int) -> list[dict]:
    if len(chunks) <= max_count or max_count <= 1:
        return chunks[:max_count] if max_count <= 1 else chunks
    step = (len(chunks) - 1) / (max_count - 1)
    indices = sorted({round(i * step) for i in range(max_count)})
    return [chunks[i] for i in indices]


def _budget_limited_sample(chunks: list[dict], max_count: int, max_chars: int) -> list[dict]:
    sample = _evenly_spaced_sample(chunks, max_count)
    total_chars = sum(len(c["text"]) for c in sample)
    if total_chars <= max_chars or not sample:
        return sample
    avg_chars = total_chars / len(sample)
    # Re-sample evenly over the already-evenly-spaced set (not the raw chunk
    # list) so the reduced set still spans the document's start and end,
    # instead of just truncating the tail off a size-sorted or index-ordered list.
    reduced_count = max(1, int(max_chars // avg_chars))
    return _evenly_spaced_sample(sample, reduced_count)


def retrieve(vector_store: VectorStore, document_ids: list[str], question: str) -> list[dict]:
    question_type = detect_question_type(question)

    if question_type == "factual":
        query_embedding = embed_texts([question])[0]
        merged = []
        for document_id in document_ids:
            for match in vector_store.query(document_id, query_embedding, k=FACTUAL_K):
                match["document_id"] = document_id
                merged.append(match)
        merged.sort(key=lambda m: m["distance"])
        return dedupe_by_id(merged)[:FACTUAL_K]

    if question_type == "summarization":
        # A representative overview of the WHOLE document reads better than
        # chunks that happen to be semantically closest to the literal words
        # "summarize this document" — that tends to concentrate on just a
        # couple of pages. Sample evenly across every chunk instead, capped
        # so large documents still fit the LLM's context window.
        sampled = []
        per_document_char_budget = SUMMARY_MAX_CONTEXT_CHARS // max(len(document_ids), 1)
        for document_id in document_ids:
            doc_chunks = vector_store.get_all_chunks(document_id)
            for chunk in doc_chunks:
                chunk["document_id"] = document_id
            sampled.extend(
                _budget_limited_sample(doc_chunks, SUMMARY_MAX_CHUNKS_PER_DOCUMENT, per_document_char_budget)
            )
        return sampled

    # exhaustive: hybrid keyword + semantic, since pure top-k can silently
    # miss occurrences a keyword scan would catch. The semantic half is
    # capped globally (BROAD_K across all documents); the keyword half is
    # kept in full so "find all references" never silently drops a real
    # match just because more documents are loaded.
    query_embedding = embed_texts([question])[0]
    semantic_results = []
    keyword_results = []
    keywords = extract_keywords(question)
    for document_id in document_ids:
        for match in vector_store.query(document_id, query_embedding, k=BROAD_K):
            match["document_id"] = document_id
            semantic_results.append(match)
        doc_chunks = vector_store.get_all_chunks(document_id)
        for chunk in doc_chunks:
            chunk["document_id"] = document_id
        keyword_results.extend(keyword_search(doc_chunks, keywords))

    semantic_results.sort(key=lambda m: m["distance"])
    return dedupe_by_id(keyword_results + semantic_results[:BROAD_K])
