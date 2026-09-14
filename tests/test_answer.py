from pathlib import Path

from core.answer import answer_question
from core.indexing import index_document
from core.prompt import NOT_FOUND_MESSAGE
from core.vectorstore import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def _indexed_store(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(str(FIXTURES / "sample.txt"), document_id="txt-doc", vector_store=store)
    return store


def test_answer_question_returns_grounded_answer_with_sources(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM("The fox jumps over the lazy dog. [lines 1-20]")

    answer = answer_question(store, "txt-doc", "What does the fox do?", llm=llm)

    assert answer.text == "The fox jumps over the lazy dog. [lines 1-20]"
    assert answer.found_in_document is True
    assert len(answer.sources) > 0
    assert all(s.location_label.startswith("lines ") for s in answer.sources)


def test_answer_question_detects_not_found_response(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM(NOT_FOUND_MESSAGE)

    answer = answer_question(store, "txt-doc", "What is the capital of France?", llm=llm)

    assert answer.found_in_document is False
    assert answer.sources == []


def test_answer_question_passes_chat_history_into_the_prompt(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM("some answer")
    history = [("user", "earlier question"), ("assistant", "earlier answer")]

    answer_question(store, "txt-doc", "a follow-up question", llm=llm, chat_history=history)

    assert "earlier question" in llm.last_prompt
    assert "earlier answer" in llm.last_prompt
