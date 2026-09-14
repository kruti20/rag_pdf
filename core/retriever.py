import re

from core.embedder import embed_texts
from core.vectorstore import VectorStore

FACTUAL_K = 5
BROAD_K = 12

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


def retrieve(vector_store: VectorStore, document_id: str, question: str) -> list[dict]:
    question_type = detect_question_type(question)
    query_embedding = embed_texts([question])[0]

    if question_type == "factual":
        results = vector_store.query(document_id, query_embedding, k=FACTUAL_K)
        return dedupe_by_id(results)

    # summarization / exhaustive: hybrid keyword + semantic, since pure top-k
    # can silently miss occurrences a keyword scan would catch.
    semantic_results = vector_store.query(document_id, query_embedding, k=BROAD_K)
    keywords = extract_keywords(question)
    keyword_results = keyword_search(vector_store.get_all_chunks(document_id), keywords)
    return dedupe_by_id(keyword_results + semantic_results)
