from groq import RateLimitError

from core.answer import answer_question

RATE_LIMIT_MESSAGE = (
    "Hit the free-tier rate limit, retrying in a few seconds… please try again shortly."
)
GENERIC_ERROR_MESSAGE = "Something went wrong answering that question. Please try again."


def build_answer_message(
    vector_store,
    document_id: str,
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    answer_fn=answer_question,
) -> dict:
    try:
        answer = answer_fn(vector_store, document_id, question, chat_history=chat_history)
    except RateLimitError:
        return {"role": "assistant", "text": RATE_LIMIT_MESSAGE, "sources": [], "found_in_document": False}
    except Exception:
        return {"role": "assistant", "text": GENERIC_ERROR_MESSAGE, "sources": [], "found_in_document": False}

    return {
        "role": "assistant",
        "text": answer.text,
        "sources": [s.location_label for s in answer.sources],
        "found_in_document": answer.found_in_document,
    }
