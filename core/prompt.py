NOT_FOUND_MESSAGE = "I couldn't find that in the document."

SYSTEM_INSTRUCTIONS = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "context excerpts provided below, each labeled with its source location. "
    "Cite the location label(s) you used in your answer. "
    "If asked to summarize, write a thorough, well-organized summary that "
    "covers every major section or topic present in the context — do not "
    "compress it into just one or two sentences. "
    f'If the answer is not covered by the provided context, respond exactly with "{NOT_FOUND_MESSAGE}" '
    "and nothing else — do not guess or use outside knowledge."
)


def _format_chunk(chunk: dict, document_names: dict[str, str]) -> str:
    name = document_names.get(chunk.get("document_id"))
    label = f"{name} — {chunk['location_label']}" if name else chunk["location_label"]
    return f"[{label}] {chunk['text']}"


def build_prompt(
    question: str,
    chunks: list[dict],
    document_names: dict[str, str] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> str:
    document_names = document_names or {}
    context = (
        "\n\n".join(_format_chunk(c, document_names) for c in chunks)
        if chunks
        else "(no relevant context found in the document)"
    )

    history_section = ""
    if chat_history:
        history_lines = "\n".join(f"{role}: {text}" for role, text in chat_history)
        history_section = f"\n\nConversation so far:\n{history_lines}"

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context}"
        f"{history_section}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
