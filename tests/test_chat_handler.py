import httpx
from groq import RateLimitError

from core.chat_handler import build_answer_message
from core.models import Answer, Source


def _rate_limit_error():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError("rate limited", response=response, body=None)


def test_successful_answer_produces_assistant_message_with_sources():
    def fake_answer_fn(vector_store, document_id, question, chat_history=None):
        return Answer(
            text="The due date is the 1st.",
            found_in_document=True,
            sources=[Source(location_label="p.1", snippet="Payment due date is the 1st.")],
        )

    message = build_answer_message(
        None, "doc1", "When is it due?", answer_fn=fake_answer_fn
    )

    assert message == {
        "role": "assistant",
        "text": "The due date is the 1st.",
        "sources": ["p.1"],
        "found_in_document": True,
    }


def test_not_found_answer_produces_message_with_no_sources():
    def fake_answer_fn(vector_store, document_id, question, chat_history=None):
        return Answer(
            text="I couldn't find that in the document.", found_in_document=False, sources=[]
        )

    message = build_answer_message(None, "doc1", "?", answer_fn=fake_answer_fn)

    assert message["found_in_document"] is False
    assert message["sources"] == []


def test_rate_limit_error_produces_friendly_retry_message():
    def failing_answer_fn(vector_store, document_id, question, chat_history=None):
        raise _rate_limit_error()

    message = build_answer_message(None, "doc1", "?", answer_fn=failing_answer_fn)

    assert message["role"] == "assistant"
    assert message["found_in_document"] is False
    assert message["sources"] == []
    assert "rate limit" in message["text"].lower()


def test_generic_error_produces_distinct_message_not_mentioning_rate_limit():
    def failing_answer_fn(vector_store, document_id, question, chat_history=None):
        raise RuntimeError("boom")

    message = build_answer_message(None, "doc1", "?", answer_fn=failing_answer_fn)

    assert message["found_in_document"] is False
    assert "rate limit" not in message["text"].lower()
    assert "went wrong" in message["text"].lower()


def test_answer_fn_is_called_with_vector_store_document_id_question_and_history():
    received = {}

    def fake_answer_fn(vector_store, document_id, question, chat_history=None):
        received["args"] = (vector_store, document_id, question, chat_history)
        return Answer(text="ok", found_in_document=True, sources=[])

    sentinel_store = object()
    build_answer_message(
        sentinel_store,
        "doc1",
        "What?",
        chat_history=[("user", "prior")],
        answer_fn=fake_answer_fn,
    )

    assert received["args"] == (sentinel_store, "doc1", "What?", [("user", "prior")])
