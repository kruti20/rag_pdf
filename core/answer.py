from core.llm import GroqLLM
from core.models import Answer, Source
from core.prompt import NOT_FOUND_MESSAGE, build_prompt
from core.retriever import retrieve
from core.vectorstore import VectorStore

SNIPPET_LENGTH = 200


def answer_question(
    vector_store: VectorStore,
    document_id: str,
    question: str,
    llm=None,
    chat_history: list[tuple[str, str]] | None = None,
) -> Answer:
    chunks = retrieve(vector_store, document_id, question)
    prompt = build_prompt(question, chunks, chat_history=chat_history)

    llm = llm or GroqLLM()
    text = llm.generate(prompt)

    found_in_document = NOT_FOUND_MESSAGE.lower() not in text.lower()
    sources = (
        [
            Source(location_label=c["location_label"], snippet=c["text"][:SNIPPET_LENGTH])
            for c in chunks
        ]
        if found_in_document
        else []
    )

    return Answer(text=text, found_in_document=found_in_document, sources=sources)
