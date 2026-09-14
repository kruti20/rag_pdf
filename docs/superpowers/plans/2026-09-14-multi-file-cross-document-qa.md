# Multi-File Cross-Document Q&A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a session hold up to 10 documents; factual/exhaustive questions search the whole library with per-source citations, while summarization questions ask which single document to summarize when more than one is loaded.

**Architecture:** `core/retriever.py` gains a document-list loop with a global-distance merge/cap so multi-document cost matches single-document cost for factual questions. `core/prompt.py`, `core/answer.py`, and `core/chat_handler.py` thread a `document_id -> name` map through so citations name their source file. `app.py` replaces the single-document upload slot with a library (upload/remove, no scope checkboxes) and adds an inline chat picker for the summarization special case.

**Tech Stack:** Python 3.11, Streamlit, ChromaDB (`chromadb.PersistentClient`), `sentence-transformers` (`all-MiniLM-L6-v2`), Groq (`openai/gpt-oss-120b`), pytest.

## Global Constraints

- Max 10 documents per session (`MAX_DOCUMENTS = 10` in `app.py`).
- Chunk IDs are `f"{document_id}-{index}"` (see `core/chunker.py:27`) — globally unique, so cross-document result merging can dedupe by ID safely.
- `FACTUAL_K = 5` and `BROAD_K = 12` (see `core/retriever.py:6-7`) become **global** caps applied after merging results across all queried documents, not per-document caps.
- No artificial cap on exhaustive ("find all references") keyword-search results — every real match across every selected document is returned, per PRD F6 ("don't miss instances").
- Summarization is always scoped to exactly one document. Ask the user which one only when more than one document is currently loaded; skip the prompt when exactly one is loaded.
- No persistence of the document library or chat history across app restarts (unchanged from existing behavior).
- The project is now a git repo (initialized this session, root commit `fc25eb9`). Commit after each task below.

---

### Task 1: VectorStore.delete_document

**Files:**
- Modify: `core/vectorstore.py`
- Test: `tests/test_vectorstore.py`

**Interfaces:**
- Consumes: `self.client` (`chromadb.PersistentClient`, already set in `__init__`), `self._collection` naming convention `f"doc_{document_id}"`.
- Produces: `VectorStore.delete_document(self, document_id: str) -> None` — used by Task 6 (app.py's 🗑 remove button).

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/test_vectorstore.py`:

```python
def test_delete_document_removes_its_chunks(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks([_chunk("doc1-0", "doc1", "p.1", "payment due date", [1.0, 0.0, 0.0])])

    store.delete_document("doc1")

    assert store.get_all_chunks("doc1") == []


def test_delete_document_does_not_affect_other_documents(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks([_chunk("docA-0", "docA", "p.1", "from doc A", [1.0, 0.0, 0.0])])
    store.add_chunks([_chunk("docB-0", "docB", "p.1", "from doc B", [1.0, 0.0, 0.0])])

    store.delete_document("docA")

    assert store.get_all_chunks("docA") == []
    assert [c["id"] for c in store.get_all_chunks("docB")] == ["docB-0"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vectorstore.py -v`
Expected: `test_delete_document_removes_its_chunks` and `test_delete_document_does_not_affect_other_documents` FAIL with `AttributeError: 'VectorStore' object has no attribute 'delete_document'`.

- [ ] **Step 3: Implement `delete_document`**

In `core/vectorstore.py`, add this method to the `VectorStore` class, directly after `_collection`:

```python
    def delete_document(self, document_id: str) -> None:
        self.client.delete_collection(name=f"doc_{document_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vectorstore.py -v`
Expected: all tests PASS (7 existing + 2 new = 9).

- [ ] **Step 5: Commit**

```bash
git add core/vectorstore.py tests/test_vectorstore.py
git commit -m "feat: add VectorStore.delete_document"
```

---

### Task 2: Cross-document retrieval with a global top-k cap

**Files:**
- Modify: `core/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `VectorStore.query(document_id, query_embedding, k) -> list[dict]`, `VectorStore.get_all_chunks(document_id) -> list[dict]` (both unchanged, from Task 1's file), `embed_texts(texts: list[str]) -> list[list[float]]` (`core/embedder.py`, unchanged).
- Produces: `retrieve(vector_store: VectorStore, document_ids: list[str], question: str) -> list[dict]`. Each returned dict now always includes a `"document_id"` key (in addition to existing `id`, `text`, `location_label`, `source_type`, and — for semantically-retrieved ones — `distance`). Used by Task 4 (`core/answer.py`).

- [ ] **Step 1: Update existing tests to the list-based signature and add cross-document tests**

Replace the two `retrieve(...)` call sites and add two new tests in `tests/test_retriever.py`. Replace lines 76-101 (the two `test_retrieve_*` functions) with:

```python
def test_retrieve_factual_question_returns_a_small_top_k(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf"),
        document_id="pdf-doc",
        vector_store=store,
    )

    results = retrieve(store, ["pdf-doc"], "What is DE Monitoring?")

    assert 0 < len(results) <= 5
    assert any(r["location_label"] == "p.4" for r in results)
    assert all(r["document_id"] == "pdf-doc" for r in results)


def test_retrieve_exhaustive_question_finds_keyword_matches_semantic_search_could_miss(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf"),
        document_id="pdf-doc",
        vector_store=store,
    )

    results = retrieve(store, ["pdf-doc"], "Find all references to prompt injection")

    assert any(r["location_label"] == "p.22" for r in results)
    assert len(results) > 5  # broader than a plain factual top-k


def test_retrieve_merges_factual_results_across_documents_and_caps_globally(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("The quarterly revenue report shows steady growth.\n" * 5)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("The lease termination clause requires 30 days notice.\n" * 5)
    index_document(str(doc_a), document_id="doc-a", vector_store=store)
    index_document(str(doc_b), document_id="doc-b", vector_store=store)

    results = retrieve(store, ["doc-a", "doc-b"], "What does the lease termination clause require?")

    assert 0 < len(results) <= 5
    assert any(r["document_id"] == "doc-b" for r in results)


def test_retrieve_exhaustive_merges_keyword_matches_across_documents(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("Penalty clause: late delivery incurs a penalty fee.\n" * 3)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("Separate penalty clause: early termination incurs a penalty fee.\n" * 3)
    index_document(str(doc_a), document_id="doc-a", vector_store=store)
    index_document(str(doc_b), document_id="doc-b", vector_store=store)

    results = retrieve(store, ["doc-a", "doc-b"], "Find all references to penalty")

    found_document_ids = {r["document_id"] for r in results}
    assert found_document_ids == {"doc-a", "doc-b"}
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `pytest tests/test_retriever.py -v`
Expected: the two existing tests FAIL (passing a string where a list is now expected causes `vector_store.query` to be called per-character, e.g. `query("p", ...)`, returning empty/wrong collections); the two new cross-document tests FAIL with `AssertionError` (e.g. `doc-b` never appears since `retrieve` doesn't yet accept a list).

- [ ] **Step 3: Implement multi-document `retrieve`**

In `core/retriever.py`, replace the `retrieve` function (the final function in the file) with:

```python
def retrieve(vector_store: VectorStore, document_ids: list[str], question: str) -> list[dict]:
    question_type = detect_question_type(question)
    query_embedding = embed_texts([question])[0]

    if question_type == "factual":
        merged = []
        for document_id in document_ids:
            for match in vector_store.query(document_id, query_embedding, k=FACTUAL_K):
                match["document_id"] = document_id
                merged.append(match)
        merged.sort(key=lambda m: m["distance"])
        return dedupe_by_id(merged)[:FACTUAL_K]

    # summarization / exhaustive: hybrid keyword + semantic, since pure top-k
    # can silently miss occurrences a keyword scan would catch. The semantic
    # half is capped globally (BROAD_K across all documents); the keyword
    # half is kept in full so "find all references" never silently drops a
    # real match just because more documents are loaded.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: all 13 tests PASS (9 existing unchanged + 2 updated + 2 new).

- [ ] **Step 5: Commit**

```bash
git add core/retriever.py tests/test_retriever.py
git commit -m "feat: retrieve across multiple documents with a global top-k cap"
```

---

### Task 3: Document-name labels in the LLM prompt

**Files:**
- Modify: `core/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: nothing new (pure string formatting).
- Produces: `build_prompt(question: str, chunks: list[dict], document_names: dict[str, str] | None = None, chat_history: list[tuple[str, str]] | None = None) -> str`. Backward compatible: `document_names` defaults to `None`/`{}`, and any chunk dict missing a recognized `"document_id"` falls back to the old bare `[location_label]` format. Used by Task 4 (`core/answer.py`).

- [ ] **Step 1: Write the failing test**

Add to the end of `tests/test_prompt.py`:

```python
def test_prompt_labels_each_chunk_with_its_document_name():
    chunks = [
        {"document_id": "doc-a", "location_label": "p.1", "text": "Payment is due on the first."},
        {"document_id": "doc-b", "location_label": "p.2", "text": "Termination requires notice."},
    ]
    document_names = {"doc-a": "ContractA.pdf", "doc-b": "ContractB.pdf"}

    prompt = build_prompt("What is the due date?", chunks, document_names)

    assert "[ContractA.pdf — p.1]" in prompt
    assert "[ContractB.pdf — p.2]" in prompt


def test_prompt_falls_back_to_bare_location_label_when_document_name_unknown():
    chunks = [{"document_id": "doc-a", "location_label": "p.1", "text": "Payment is due on the first."}]

    prompt = build_prompt("What is the due date?", chunks)

    assert "[p.1]" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prompt.py -v`
Expected: `test_prompt_labels_each_chunk_with_its_document_name` FAILS (prompt contains `[p.1]` and `[p.2]`, not the name-prefixed form); `test_prompt_falls_back_to_bare_location_label_when_document_name_unknown` PASSES already (no behavior change needed for that case) — confirm it passes for the right reason once Step 3 lands.

- [ ] **Step 3: Implement document-name labeling**

In `core/prompt.py`, replace `_format_chunk` and `build_prompt` with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prompt.py -v`
Expected: all 8 tests PASS (6 existing unchanged + 2 new).

- [ ] **Step 5: Commit**

```bash
git add core/prompt.py tests/test_prompt.py
git commit -m "feat: label prompt context chunks with their source document name"
```

---

### Task 4: Cross-document answers with per-source document IDs

**Files:**
- Modify: `core/models.py`
- Modify: `core/answer.py`
- Test: `tests/test_answer.py`

**Interfaces:**
- Consumes: `retrieve(vector_store, document_ids, question)` (Task 2), `build_prompt(question, chunks, document_names, chat_history)` (Task 3), `GroqLLM` (unchanged, `core/llm.py`).
- Produces: `Source` dataclass gains `document_id: str = ""` (default keeps old call sites like `Source(location_label=..., snippet=...)` valid). `answer_question(vector_store, document_ids: list[str], question: str, document_names: dict[str, str] | None = None, llm=None, chat_history: list[tuple[str, str]] | None = None) -> Answer`. Used by Task 5 (`core/chat_handler.py`).

- [ ] **Step 1: Write the failing tests**

In `core/models.py`, this task will add `document_id` to `Source` — no separate test file for that (it's a plain dataclass field, exercised via `test_answer.py`).

Replace `tests/test_answer.py` in full with:

```python
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

    answer = answer_question(store, ["txt-doc"], "What does the fox do?", llm=llm)

    assert answer.text == "The fox jumps over the lazy dog. [lines 1-20]"
    assert answer.found_in_document is True
    assert len(answer.sources) > 0
    assert all(s.location_label.startswith("lines ") for s in answer.sources)
    assert all(s.document_id == "txt-doc" for s in answer.sources)


def test_answer_question_detects_not_found_response(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM(NOT_FOUND_MESSAGE)

    answer = answer_question(store, ["txt-doc"], "What is the capital of France?", llm=llm)

    assert answer.found_in_document is False
    assert answer.sources == []


def test_answer_question_passes_chat_history_into_the_prompt(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM("some answer")
    history = [("user", "earlier question"), ("assistant", "earlier answer")]

    answer_question(store, ["txt-doc"], "a follow-up question", llm=llm, chat_history=history)

    assert "earlier question" in llm.last_prompt
    assert "earlier answer" in llm.last_prompt


def test_answer_question_citations_span_multiple_documents(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("The quarterly revenue report shows steady growth.\n" * 3)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("The lease termination clause requires 30 days notice.\n" * 3)
    index_document(str(doc_a), document_id="doc-a", vector_store=store)
    index_document(str(doc_b), document_id="doc-b", vector_store=store)
    llm = _FakeLLM("The lease requires 30 days notice. [Lease.txt — lines 1-3]")

    answer = answer_question(
        store,
        ["doc-a", "doc-b"],
        "What does the lease termination clause require?",
        document_names={"doc-a": "Revenue.txt", "doc-b": "Lease.txt"},
        llm=llm,
    )

    assert any(s.document_id == "doc-b" for s in answer.sources)
    assert "Lease.txt" in llm.last_prompt
    assert "Revenue.txt" in llm.last_prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_answer.py -v`
Expected: FAIL — `answer_question` still takes a single `document_id: str`, so passing a list breaks retrieval (per Task 2's new signature), and `Source` has no `document_id` attribute yet.

- [ ] **Step 3: Implement `Source.document_id` and multi-document `answer_question`**

In `core/models.py`, update the `Source` dataclass:

```python
@dataclass
class Source:
    location_label: str
    snippet: str
    document_id: str = ""
```

In `core/answer.py`, replace `answer_question` with:

```python
def answer_question(
    vector_store: VectorStore,
    document_ids: list[str],
    question: str,
    document_names: dict[str, str] | None = None,
    llm=None,
    chat_history: list[tuple[str, str]] | None = None,
) -> Answer:
    document_names = document_names or {}
    chunks = retrieve(vector_store, document_ids, question)
    prompt = build_prompt(question, chunks, document_names=document_names, chat_history=chat_history)

    llm = llm or GroqLLM()
    text = llm.generate(prompt)

    found_in_document = NOT_FOUND_MESSAGE.lower() not in text.lower()
    sources = (
        [
            Source(
                location_label=c["location_label"],
                snippet=c["text"][:SNIPPET_LENGTH],
                document_id=c.get("document_id", ""),
            )
            for c in chunks
        ]
        if found_in_document
        else []
    )

    return Answer(text=text, found_in_document=found_in_document, sources=sources)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_answer.py -v`
Expected: all 4 tests PASS.

Also run the full suite to confirm nothing upstream broke:

Run: `pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/models.py core/answer.py tests/test_answer.py
git commit -m "feat: thread document_ids and document_names through answer_question"
```

---

### Task 5: Citations formatted with document names in chat messages

**Files:**
- Modify: `core/chat_handler.py`
- Test: `tests/test_chat_handler.py`

**Interfaces:**
- Consumes: `answer_question(vector_store, document_ids, question, document_names, llm, chat_history)` (Task 4, as the default `answer_fn`), `Source.document_id` (Task 4).
- Produces: `build_answer_message(vector_store, document_ids: list[str], question: str, document_names: dict[str, str] | None = None, chat_history: list[tuple[str, str]] | None = None, answer_fn=answer_question) -> dict`. The returned dict's `"sources"` list contains strings like `"ContractA.pdf — p.1"` when the document name is known, or bare `"p.1"` otherwise. Used by Task 6 (`app.py`).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_chat_handler.py` in full with:

```python
import httpx
from groq import RateLimitError

from core.chat_handler import build_answer_message
from core.models import Answer, Source


def _rate_limit_error():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError("rate limited", response=response, body=None)


def test_successful_answer_produces_assistant_message_with_named_sources():
    def fake_answer_fn(vector_store, document_ids, question, document_names=None, chat_history=None):
        return Answer(
            text="The due date is the 1st.",
            found_in_document=True,
            sources=[Source(location_label="p.1", snippet="Payment due date is the 1st.", document_id="doc1")],
        )

    message = build_answer_message(
        None, ["doc1"], "When is it due?", document_names={"doc1": "Contract.pdf"}, answer_fn=fake_answer_fn
    )

    assert message == {
        "role": "assistant",
        "text": "The due date is the 1st.",
        "sources": ["Contract.pdf — p.1"],
        "found_in_document": True,
    }


def test_source_without_a_known_document_name_falls_back_to_bare_location_label():
    def fake_answer_fn(vector_store, document_ids, question, document_names=None, chat_history=None):
        return Answer(
            text="The due date is the 1st.",
            found_in_document=True,
            sources=[Source(location_label="p.1", snippet="...", document_id="doc1")],
        )

    message = build_answer_message(None, ["doc1"], "When is it due?", answer_fn=fake_answer_fn)

    assert message["sources"] == ["p.1"]


def test_not_found_answer_produces_message_with_no_sources():
    def fake_answer_fn(vector_store, document_ids, question, document_names=None, chat_history=None):
        return Answer(
            text="I couldn't find that in the document.", found_in_document=False, sources=[]
        )

    message = build_answer_message(None, ["doc1"], "?", answer_fn=fake_answer_fn)

    assert message["found_in_document"] is False
    assert message["sources"] == []


def test_rate_limit_error_produces_friendly_retry_message():
    def failing_answer_fn(vector_store, document_ids, question, document_names=None, chat_history=None):
        raise _rate_limit_error()

    message = build_answer_message(None, ["doc1"], "?", answer_fn=failing_answer_fn)

    assert message["role"] == "assistant"
    assert message["found_in_document"] is False
    assert message["sources"] == []
    assert "rate limit" in message["text"].lower()


def test_generic_error_produces_distinct_message_not_mentioning_rate_limit():
    def failing_answer_fn(vector_store, document_ids, question, document_names=None, chat_history=None):
        raise RuntimeError("boom")

    message = build_answer_message(None, ["doc1"], "?", answer_fn=failing_answer_fn)

    assert message["found_in_document"] is False
    assert "rate limit" not in message["text"].lower()
    assert "went wrong" in message["text"].lower()


def test_answer_fn_is_called_with_vector_store_document_ids_names_and_history():
    received = {}

    def fake_answer_fn(vector_store, document_ids, question, document_names=None, chat_history=None):
        received["args"] = (vector_store, document_ids, question, document_names, chat_history)
        return Answer(text="ok", found_in_document=True, sources=[])

    sentinel_store = object()
    build_answer_message(
        sentinel_store,
        ["doc1"],
        "What?",
        document_names={"doc1": "Contract.pdf"},
        chat_history=[("user", "prior")],
        answer_fn=fake_answer_fn,
    )

    assert received["args"] == (
        sentinel_store,
        ["doc1"],
        "What?",
        {"doc1": "Contract.pdf"},
        [("user", "prior")],
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat_handler.py -v`
Expected: FAIL — `build_answer_message` doesn't yet accept `document_names`, and `"sources"` is still bare `location_label` strings (`"p.1"` instead of `"Contract.pdf — p.1"`).

- [ ] **Step 3: Implement named citations**

Replace `core/chat_handler.py` in full with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_handler.py -v`
Expected: all 6 tests PASS.

Also run the full suite:

Run: `pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add core/chat_handler.py tests/test_chat_handler.py
git commit -m "feat: format chat citations with their source document name"
```

---

### Task 6: Multi-document library UI with the summarize picker

**Files:**
- Modify: `app.py`
- Modify: `run_steps.txt`

**Interfaces:**
- Consumes: `VectorStore.delete_document` (Task 1), `retrieve`'s `detect_question_type` re-export (`core/retriever.py`, unchanged since original code — signature `detect_question_type(question: str) -> str`, returns `"factual" | "exhaustive" | "summarization"`), `build_answer_message(vector_store, document_ids, question, document_names, chat_history, answer_fn)` (Task 5), `index_document`, `DocumentValidationError`, `VectorStore` (all unchanged).
- Produces: the running app. Terminal task — nothing downstream consumes this.

- [ ] **Step 1: Replace `app.py` in full**

```python
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.chat_handler import build_answer_message
from core.indexing import index_document
from core.ingest import DocumentValidationError
from core.retriever import detect_question_type
from core.vectorstore import VectorStore

st.set_page_config(page_title="AI PDF Document Assistant", page_icon="📄", layout="wide")

MAX_DOCUMENTS = 10

SUGGESTED_QUESTIONS = [
    "Summarize this document",
    "What are the key dates?",
    "List any obligations or penalties",
]

SIZE_LABELS = {"pdf": "pages with text", "docx": "paragraphs", "txt": "sections"}


@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()


def reset_chat():
    st.session_state.chat_history = []


if "documents" not in st.session_state:
    st.session_state.documents = {}
    reset_chat()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "pending_summarize_question" not in st.session_state:
    st.session_state.pending_summarize_question = None

st.title("📄 AI PDF Document Assistant")

left, right = st.columns([1, 2])

with left:
    st.subheader(f"Documents ({len(st.session_state.documents)}/{MAX_DOCUMENTS})")
    uploaded_files = st.file_uploader(
        "Drag PDF, Word (.docx), or text files here", type=None, accept_multiple_files=True
    )

    for uploaded_file in uploaded_files or []:
        file_bytes = uploaded_file.getvalue()
        document_id = hashlib.sha256(file_bytes).hexdigest()[:16]

        if document_id in st.session_state.documents:
            continue

        if len(st.session_state.documents) >= MAX_DOCUMENTS:
            st.error(
                f"Maximum {MAX_DOCUMENTS} documents — remove one before adding "
                f"'{uploaded_file.name}'"
            )
            continue

        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                result = index_document(
                    tmp_path, document_id=document_id, vector_store=get_vector_store()
                )
        except DocumentValidationError as e:
            st.error(f"{uploaded_file.name}: {e}")
        else:
            st.session_state.documents[document_id] = {
                "name": uploaded_file.name,
                "source_type": result.source_type,
                "section_count": len(result.segments),
                "chunk_count": result.chunk_count,
                "has_extractable_text": result.has_extractable_text,
            }

    for document_id, meta in list(st.session_state.documents.items()):
        with st.container(border=True):
            st.markdown(f"📄 **{meta['name']}**")
            size_label = SIZE_LABELS.get(meta["source_type"], "sections")
            st.caption(f"{meta['section_count']} {size_label} · {meta['source_type'].upper()}")

            if not meta["has_extractable_text"]:
                st.warning(
                    "This document looks scanned or has no extractable text — "
                    "answers may be limited or unavailable."
                )
            else:
                st.success(f"indexed — {meta['chunk_count']} chunks")

            if st.button("🗑 Remove", key=f"remove-{document_id}"):
                get_vector_store().delete_document(document_id)
                del st.session_state.documents[document_id]
                st.rerun()

    if st.session_state.documents:
        st.markdown("**Suggested questions:**")
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, key=f"suggested-{question}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

with right:
    st.subheader("Chat")

    if not st.session_state.documents:
        st.info("Upload a PDF, Word (.docx), or text file to get started.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and not message.get("found_in_document", True):
                st.markdown(f"*{message['text']}*")
            else:
                st.markdown(message["text"])
                if message.get("sources"):
                    st.caption("Sources: " + ", ".join(message["sources"]))

    if st.session_state.pending_summarize_question and not st.session_state.documents:
        st.session_state.pending_summarize_question = None

    if st.session_state.pending_summarize_question:
        with st.chat_message("assistant"):
            st.markdown("Which document would you like to summarize?")
            library = list(st.session_state.documents.items())
            choice_name = st.radio(
                "Choose a document",
                options=[meta["name"] for _, meta in library],
                key="summarize-choice",
                label_visibility="collapsed",
            )
            if st.button("Summarize", key="summarize-confirm"):
                chosen_id = next(doc_id for doc_id, meta in library if meta["name"] == choice_name)
                question = st.session_state.pending_summarize_question
                document_names = {doc_id: meta["name"] for doc_id, meta in st.session_state.documents.items()}
                history_for_prompt = [
                    (m["role"], m["text"]) for m in st.session_state.chat_history[:-1]
                ]
                with st.spinner("Summarizing…"):
                    message = build_answer_message(
                        get_vector_store(),
                        [chosen_id],
                        question,
                        document_names=document_names,
                        chat_history=history_for_prompt,
                    )
                    st.session_state.chat_history.append(message)
                st.session_state.pending_summarize_question = None
                st.rerun()

    documents_ready = bool(st.session_state.documents) and not st.session_state.pending_summarize_question
    question = st.chat_input(
        "Ask a question about your documents" if st.session_state.documents else "Upload a document to get started",
        disabled=not documents_ready,
    )
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question and documents_ready:
        st.session_state.chat_history.append({"role": "user", "text": question})
        document_ids = list(st.session_state.documents.keys())
        document_names = {doc_id: meta["name"] for doc_id, meta in st.session_state.documents.items()}

        if detect_question_type(question) == "summarization" and len(document_ids) > 1:
            st.session_state.pending_summarize_question = question
            st.rerun()
        else:
            history_for_prompt = [
                (m["role"], m["text"]) for m in st.session_state.chat_history[:-1]
            ]
            with st.spinner("Searching documents…"):
                message = build_answer_message(
                    get_vector_store(),
                    document_ids,
                    question,
                    document_names=document_names,
                    chat_history=history_for_prompt,
                )
                st.session_state.chat_history.append(message)
            st.rerun()
```

- [ ] **Step 2: Update `run_steps.txt`**

In the "6. Use the app" section, replace:

```
   - Upload a PDF, DOCX, or TXT file (max 20MB; legacy .doc is rejected - save as .docx)
   - Wait for "indexed - N chunks" to appear
   - Ask a question in the chat box, or click one of the suggested questions
   - Answers are grounded in the document and cited by page/paragraph/line
```

with:

```
   - Upload up to 10 PDF, DOCX, or TXT files (max 20MB each; legacy .doc is rejected - save as .docx)
   - Wait for "indexed - N chunks" to appear next to each file
   - Ask a question in the chat box, or click one of the suggested questions
   - Questions search across every uploaded document; answers are grounded and
     cited by document name plus page/paragraph/line
   - Asking to "summarize" with more than one document loaded will prompt you
     to pick which single document to summarize
   - Remove a document with its 🗑 button to free a slot (max 10 at a time)
```

In "Known limitations (MVP)", replace:

```
- Single-document sessions only (no cross-document Q&A)
```

with:

```
- Up to 10 documents per session; the library resets on restart (same as chat history)
- "Find all references to X" across many large documents can be slow/costly, since
  every real match is returned rather than capped
```

- [ ] **Step 3: Manually verify the app**

Run: `streamlit run app.py`

Verify each of the following in the browser (usually `http://localhost:8501`):

1. Upload two small files (e.g. a `.txt` and the `sample.txt`-style fixture, or any two real documents). Confirm both show "indexed — N chunks" in their own card under "Documents (2/10)".
2. Ask a factual question whose answer lives in only one of the two documents. Confirm the citation caption includes that document's filename (e.g. "Contract.pdf — p.1"), not just the bare page label.
3. Ask "Summarize this document" (or click the suggested "Summarize this document" button) with both documents loaded. Confirm a picker appears in the chat ("Which document would you like to summarize?") with a radio of both filenames and a "Summarize" button, and that the chat input is disabled until you pick one and confirm.
4. Remove one document via its 🗑 button. Confirm the count updates (e.g. "Documents (1/10)") and a follow-up "Summarize this document" question now answers immediately with no picker (since only one document remains).
5. Upload files until the library is at 10, then attempt an 11th. Confirm the error "Maximum 10 documents — remove one before adding '<name>'" appears and the 11th file is not indexed.
6. Upload a `.doc` (legacy) or oversized file. Confirm the existing per-file validation error still appears, scoped to that filename, without blocking other files uploaded in the same batch.

Record the outcome of this manual pass in your task completion notes; this task has no automated test file for `app.py`, consistent with the rest of the project.

- [ ] **Step 4: Run the full automated test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py run_steps.txt
git commit -m "feat: multi-document library with cross-document Q&A and summarize picker"
```
