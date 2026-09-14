from pathlib import Path

from core.indexing import index_document
from core.retriever import dedupe_by_id, detect_question_type, extract_keywords, keyword_search, retrieve
from core.vectorstore import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"


def test_detects_exhaustive_search_questions():
    assert detect_question_type("Find all references to penalties") == "exhaustive"
    assert detect_question_type("List all obligations in this contract") == "exhaustive"
    assert detect_question_type("What are every mention of late fees?") == "exhaustive"
    assert detect_question_type("Give me all instances of termination") == "exhaustive"


def test_detects_summarization_questions():
    assert detect_question_type("Summarize the termination conditions") == "summarization"
    assert detect_question_type("Can you give me a summary of section 4?") == "summarization"


def test_detects_factual_questions():
    assert detect_question_type("What is the payment due date?") == "factual"
    assert detect_question_type("Who is the counterparty?") == "factual"


def test_extract_keywords_strips_trigger_phrase_and_stopwords():
    assert extract_keywords("Find all references to penalties") == ["penalties"]
    assert extract_keywords("List all obligations in this contract") == ["obligations"]
    assert extract_keywords("Give me all instances of termination") == ["termination"]


def test_extract_keywords_keeps_multiple_significant_words():
    assert extract_keywords("What are every mention of late fees?") == ["late", "fees"]


def test_extract_keywords_on_plain_factual_question_returns_significant_words():
    keywords = extract_keywords("What is the payment due date?")
    assert "payment" in keywords
    assert "date" in keywords
    assert "what" not in keywords
    assert "the" not in keywords


def test_keyword_search_matches_case_insensitive_substring():
    chunks = [
        {"id": "c1", "text": "A Penalty applies for late delivery.", "location_label": "p.1"},
        {"id": "c2", "text": "Termination requires 30 days notice.", "location_label": "p.2"},
        {"id": "c3", "text": "Late penalties accrue daily.", "location_label": "p.3"},
    ]

    matches = keyword_search(chunks, ["penalty", "penalties"])

    assert {m["id"] for m in matches} == {"c1", "c3"}


def test_keyword_search_with_no_keywords_returns_empty():
    chunks = [{"id": "c1", "text": "anything", "location_label": "p.1"}]
    assert keyword_search(chunks, []) == []


def test_dedupe_by_id_preserves_first_occurrence_order():
    chunks = [
        {"id": "c1", "text": "first"},
        {"id": "c2", "text": "second"},
        {"id": "c1", "text": "first duplicate"},
        {"id": "c3", "text": "third"},
    ]

    result = dedupe_by_id(chunks)

    assert [c["id"] for c in result] == ["c1", "c2", "c3"]
    assert result[0]["text"] == "first"


def test_retrieve_factual_question_returns_a_small_top_k(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf"),
        document_id="pdf-doc",
        vector_store=store,
    )

    results = retrieve(store, "pdf-doc", "What is DE Monitoring?")

    assert 0 < len(results) <= 5
    assert any(r["location_label"] == "p.4" for r in results)


def test_retrieve_exhaustive_question_finds_keyword_matches_semantic_search_could_miss(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf"),
        document_id="pdf-doc",
        vector_store=store,
    )

    results = retrieve(store, "pdf-doc", "Find all references to prompt injection")

    assert any(r["location_label"] == "p.22" for r in results)
    assert len(results) > 5  # broader than a plain factual top-k
