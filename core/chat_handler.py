from groq import RateLimitError

from core.answer import answer_question

RATE_LIMIT_MESSAGE = (
    "Hit the free-tier rate limit, retrying in a few seconds… please try again shortly."
)
GENERIC_ERROR_MESSAGE = "Something went wrong answering that question. Please try again."


def _format_source(source, document_names: dict[str, str]) -> str:
    name = document_names.get(source.document_id)
    return f"{name} — {source.location_label}" if name else source.location_label


def build_answer_message(
    vector_store,
    document_ids: list[str],
    question: str,
    document_names: dict[str, str] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
    answer_fn=answer_question,
) -> dict:
    document_names = document_names or {}
    try:
        answer = answer_fn(
            vector_store,
            document_ids,
            question,
            document_names=document_names,
            chat_history=chat_history,
        )
    except RateLimitError:
        return {"role": "assistant", "text": RATE_LIMIT_MESSAGE, "sources": [], "found_in_document": False}
    except Exception:
        return {"role": "assistant", "text": GENERIC_ERROR_MESSAGE, "sources": [], "found_in_document": False}

    return {
        "role": "assistant",
        "text": answer.text,
        "sources": [_format_source(s, document_names) for s in answer.sources],
        "found_in_document": answer.found_in_document,
    }
