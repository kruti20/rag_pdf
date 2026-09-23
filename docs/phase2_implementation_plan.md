# Phase 2 (Production Hardening + RAG Quality) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix a real multi-visitor data-isolation bug in the deployed app, add automatic OCR for scanned PDFs, add cross-encoder reranking to factual retrieval, and produce a curated RAG evaluation report — the four items from `docs/06_PRD_PHASE2.md`.

**Architecture:** `core/vectorstore.py` gains a `session_id` parameter on every method so collections are named `doc_{session_id}_{document_id}`, isolating each browser session's data within the one shared Chroma DB. That `session_id` threads through `core/indexing.py`, `core/retriever.py`, `core/answer.py`, `core/chat_handler.py`, and `app.py` — the same shape `document_id`/`document_ids` already thread through. A new `core/ocr.py` gives `core/ingest.py` a per-page Tesseract fallback when normal text extraction is empty. A new `core/reranker.py` (a `sentence-transformers` `CrossEncoder`) re-scores a widened candidate pool in `core/retriever.py`'s factual path before the final top-5 is chosen. A new `eval/` package holds a curated question set, pure scoring functions, and an orchestration script that produces a committed `eval/report.md`.

**Tech Stack:** Python 3.11, Streamlit, ChromaDB (`chromadb.PersistentClient`), `sentence-transformers` (`all-MiniLM-L6-v2` for embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking), `pytesseract` + system Tesseract binary, Groq (`openai/gpt-oss-120b`), pytest.

## Global Constraints

- Every `VectorStore` method takes `session_id` as its first parameter (after `self`); collection names are `f"doc_{session_id}_{document_id}"`.
- `session_id` threads positionally as the second argument (right after `vector_store`) through `index_document`, `retrieve`, `answer_question`, and `build_answer_message` — matching the shape `document_ids` already has in this codebase.
- No cleanup job for orphaned per-session collections — they persist until the Streamlit Cloud process restarts (its filesystem is already ephemeral). Not solved this phase, per the PRD.
- Reranking (`core/reranker.py`) only applies to the factual retrieval path. The exhaustive/summarization hybrid path in `core/retriever.py` is unchanged.
- `FACTUAL_K = 5` stays the final result count for factual questions; a new `FACTUAL_CANDIDATE_K = 12` is the pre-rerank candidate pool width per document.
- OCR (`core/ocr.py`) only runs per-page when `page.get_text().strip()` is empty. A text-based PDF never invokes it. OCR failure must degrade to no text for that page (existing "looks scanned" warning), never crash ingestion.
- The evaluation suite (`eval/`) is not wired into the default `pytest tests/` run — it makes real Groq API calls. It is run manually via `python -m eval.run_eval`.
- Task order matters: Tasks 1-5 (session isolation) must land before Task 9 (reranking integration) touches `core/retriever.py`, since Task 9's diff is written against the session-isolated version of that file. Follow the numbered order.
- **Prerequisite for Tasks 6-7 (OCR):** install the Tesseract OCR binary locally before running their tests.
  - Windows: download the installer from https://github.com/UB-Mannheim/tesseract/wiki and install it. If `tesseract` isn't automatically on `PATH` afterward, set an environment variable `TESSERACT_CMD` to the full path of `tesseract.exe` (e.g. `C:\Program Files\Tesseract-OCR\tesseract.exe`) — `core/ocr.py` (Task 6) reads this automatically.
  - macOS: `brew install tesseract`
  - Linux (local dev): `sudo apt-get install tesseract-ocr`
  - Streamlit Cloud deployment needs no manual step here — Task 7 adds a `packages.txt` entry so Streamlit Cloud's own apt layer installs it automatically.

---

### Task 1: Session-scoped VectorStore collections

**Files:**
- Modify: `core/vectorstore.py`
- Test: `tests/test_vectorstore.py`

**Interfaces:**
- Consumes: `self.client` (`chromadb.PersistentClient`), `chromadb.errors.NotFoundError` (already imported via `chromadb`).
- Produces: every `VectorStore` method now requires `session_id: str` as its first argument — `add_chunks(session_id, chunks)`, `query(session_id, document_id, query_embedding, k=5)`, `get_all_chunks(session_id, document_id)`, `delete_document(session_id, document_id)`. Used by Task 2 (`index_document`) and Task 3 (`retrieve`).

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_vectorstore.py` with:

```python
from core.models import Chunk
from core.vectorstore import VectorStore

SESSION = "test-session"


def _chunk(id, document_id, location_label, text, embedding):
    return Chunk(
        id=id,
        document_id=document_id,
        source_type="pdf",
        location_label=location_label,
        text=text,
        embedding=embedding,
    )


def test_query_returns_the_closest_chunk_first(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    chunks = [
        _chunk("doc1-0", "doc1", "p.1", "payment due date", [1.0, 0.0, 0.0]),
        _chunk("doc1-1", "doc1", "p.2", "termination clause", [0.0, 1.0, 0.0]),
    ]
    store.add_chunks(SESSION, chunks)

    results = store.query(SESSION, "doc1", query_embedding=[1.0, 0.0, 0.0], k=2)

    assert results[0]["id"] == "doc1-0"
    assert results[0]["location_label"] == "p.1"
    assert results[0]["text"] == "payment due date"
    assert results[0]["source_type"] == "pdf"


def test_query_respects_k_limit(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    chunks = [
        _chunk(f"doc1-{i}", "doc1", f"p.{i}", f"chunk {i}", [float(i), 0.0, 0.0])
        for i in range(5)
    ]
    store.add_chunks(SESSION, chunks)

    results = store.query(SESSION, "doc1", query_embedding=[0.0, 0.0, 0.0], k=2)

    assert len(results) == 2


def test_documents_are_isolated_by_collection(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks(SESSION, [_chunk("docA-0", "docA", "p.1", "from doc A", [1.0, 0.0, 0.0])])
    store.add_chunks(SESSION, [_chunk("docB-0", "docB", "p.1", "from doc B", [1.0, 0.0, 0.0])])

    results = store.query(SESSION, "docA", query_embedding=[1.0, 0.0, 0.0], k=5)

    assert [r["id"] for r in results] == ["docA-0"]


def test_add_chunks_with_empty_list_is_a_no_op(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks(SESSION, [])  # must not raise


def test_get_all_chunks_returns_every_chunk_for_a_document(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks(
        SESSION,
        [
            _chunk("doc1-0", "doc1", "p.1", "penalty clause one", [1.0, 0.0, 0.0]),
            _chunk("doc1-1", "doc1", "p.2", "termination clause", [0.0, 1.0, 0.0]),
            _chunk("doc1-2", "doc1", "p.3", "penalty clause two", [0.0, 0.0, 1.0]),
        ],
    )

    all_chunks = store.get_all_chunks(SESSION, "doc1")

    assert {c["id"] for c in all_chunks} == {"doc1-0", "doc1-1", "doc1-2"}
    by_id = {c["id"]: c for c in all_chunks}
    assert by_id["doc1-0"]["text"] == "penalty clause one"
    assert by_id["doc1-0"]["location_label"] == "p.1"
    assert by_id["doc1-0"]["source_type"] == "pdf"


def test_get_all_chunks_on_empty_document_returns_empty_list(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    assert store.get_all_chunks(SESSION, "nonexistent-doc") == []


def test_delete_document_removes_its_chunks(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks(SESSION, [_chunk("doc1-0", "doc1", "p.1", "payment due date", [1.0, 0.0, 0.0])])

    store.delete_document(SESSION, "doc1")

    assert store.get_all_chunks(SESSION, "doc1") == []


def test_delete_document_does_not_affect_other_documents(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks(SESSION, [_chunk("docA-0", "docA", "p.1", "from doc A", [1.0, 0.0, 0.0])])
    store.add_chunks(SESSION, [_chunk("docB-0", "docB", "p.1", "from doc B", [1.0, 0.0, 0.0])])

    store.delete_document(SESSION, "docA")

    assert store.get_all_chunks(SESSION, "docA") == []
    assert [c["id"] for c in store.get_all_chunks(SESSION, "docB")] == ["docB-0"]


def test_delete_document_on_a_document_with_no_indexed_chunks_does_not_raise(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    store.delete_document(SESSION, "never-indexed-doc")  # must not raise


def test_same_document_id_under_different_sessions_does_not_collide(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks("session-a", [_chunk("doc1-0", "doc1", "p.1", "from session A", [1.0, 0.0, 0.0])])
    store.add_chunks("session-b", [_chunk("doc1-0", "doc1", "p.1", "from session B", [0.0, 1.0, 0.0])])

    results_a = store.get_all_chunks("session-a", "doc1")
    results_b = store.get_all_chunks("session-b", "doc1")

    assert [c["text"] for c in results_a] == ["from session A"]
    assert [c["text"] for c in results_b] == ["from session B"]


def test_delete_document_in_one_session_does_not_affect_same_document_id_in_another_session(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    store.add_chunks("session-a", [_chunk("doc1-0", "doc1", "p.1", "from session A", [1.0, 0.0, 0.0])])
    store.add_chunks("session-b", [_chunk("doc1-0", "doc1", "p.1", "from session B", [1.0, 0.0, 0.0])])

    store.delete_document("session-a", "doc1")

    assert store.get_all_chunks("session-a", "doc1") == []
    assert [c["text"] for c in store.get_all_chunks("session-b", "doc1")] == ["from session B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vectorstore.py -v`
Expected: FAIL — `TypeError: VectorStore.add_chunks() takes 2 positional arguments but 3 were given` (and similar for `query`/`get_all_chunks`/`delete_document`), since the current methods don't accept `session_id` yet.

- [ ] **Step 3: Implement session-scoped collections**

Replace the entire contents of `core/vectorstore.py` with:

```python
import chromadb

from core.models import Chunk

DEFAULT_PERSIST_DIR = "storage/chroma_db"


class VectorStore:
    def __init__(self, persist_directory: str = DEFAULT_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_directory)

    def _collection(self, session_id: str, document_id: str):
        return self.client.get_or_create_collection(name=f"doc_{session_id}_{document_id}")

    def delete_document(self, session_id: str, document_id: str) -> None:
        try:
            self.client.delete_collection(name=f"doc_{session_id}_{document_id}")
        except chromadb.errors.NotFoundError:
            pass

    def add_chunks(self, session_id: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        collection = self._collection(session_id, chunks[0].document_id)
        collection.add(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"location_label": c.location_label, "source_type": c.source_type}
                for c in chunks
            ],
        )

    def get_all_chunks(self, session_id: str, document_id: str) -> list[dict]:
        collection = self._collection(session_id, document_id)
        results = collection.get()

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            chunks.append(
                {
                    "id": chunk_id,
                    "text": results["documents"][i],
                    "location_label": metadata["location_label"],
                    "source_type": metadata["source_type"],
                }
            )
        return chunks

    def query(self, session_id: str, document_id: str, query_embedding: list[float], k: int = 5) -> list[dict]:
        collection = self._collection(session_id, document_id)
        results = collection.query(query_embeddings=[query_embedding], n_results=k)

        matches = []
        for i, chunk_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            matches.append(
                {
                    "id": chunk_id,
                    "text": results["documents"][0][i],
                    "location_label": metadata["location_label"],
                    "source_type": metadata["source_type"],
                    "distance": results["distances"][0][i],
                }
            )
        return matches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vectorstore.py -v`
Expected: all PASS (11 tests — 9 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add core/vectorstore.py tests/test_vectorstore.py
git commit -m "feat: scope VectorStore collections per session_id"
```

---

### Task 2: Thread session_id through index_document

**Files:**
- Modify: `core/indexing.py`
- Test: `tests/test_indexing.py`

**Interfaces:**
- Consumes: `VectorStore.add_chunks(session_id, chunks)` (Task 1).
- Produces: `index_document(file_path, document_id, vector_store, session_id) -> IngestResult` — used by Task 3's test setup, Task 4's test setup, Task 11's eval orchestration, and `app.py` (Task 5).

- [ ] **Step 1: Update the failing tests**

Replace the entire contents of `tests/test_indexing.py` with:

```python
from pathlib import Path

from core.embedder import embed_texts
from core.indexing import index_document
from core.vectorstore import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
SESSION = "test-session"


def test_index_document_wires_ingest_chunk_embed_store(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    result = index_document(
        str(FIXTURES / "sample.txt"), document_id="txt-doc", vector_store=store, session_id=SESSION
    )

    assert result.has_extractable_text is True
    query_embedding = embed_texts(["The quick brown fox jumps over the lazy dog."])[0]
    matches = store.query(SESSION, "txt-doc", query_embedding=query_embedding, k=1)
    assert len(matches) == 1
    assert matches[0]["location_label"].startswith("lines ")


def test_index_document_reports_chunk_count(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))

    result = index_document(
        str(FIXTURES / "sample.txt"), document_id="txt-doc", vector_store=store, session_id=SESSION
    )

    # sample.txt has 45 lines -> 3 segments of <=20 lines, none long enough to split further
    assert result.chunk_count == 3


def test_index_document_scanned_pdf_reports_zero_chunks(tmp_path):
    import pymupdf

    scanned_path = tmp_path / "scanned.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(str(scanned_path))
    pdf.close()

    store = VectorStore(persist_directory=str(tmp_path))
    result = index_document(
        str(scanned_path), document_id="scanned-doc", vector_store=store, session_id=SESSION
    )

    assert result.has_extractable_text is False
    assert result.chunk_count == 0


def test_query_known_phrase_from_real_pdf_returns_correct_page(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf"),
        document_id="pdf-doc",
        vector_store=store,
        session_id=SESSION,
    )

    query_embedding = embed_texts(["What are prompt injection attacks?"])[0]
    matches = store.query(SESSION, "pdf-doc", query_embedding=query_embedding, k=3)

    assert any(m["location_label"] == "p.22" for m in matches)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_indexing.py -v`
Expected: FAIL — `TypeError: index_document() got an unexpected keyword argument 'session_id'`.

- [ ] **Step 3: Implement**

Replace the entire contents of `core/indexing.py` with:

```python
from dataclasses import replace

from core.chunker import chunk_segments
from core.embedder import embed_texts
from core.ingest import IngestResult, ingest_document
from core.vectorstore import VectorStore


def index_document(
    file_path: str, document_id: str, vector_store: VectorStore, session_id: str
) -> IngestResult:
    result = ingest_document(file_path)
    if not result.has_extractable_text:
        return result

    chunks = chunk_segments(result.segments, document_id=document_id, source_type=result.source_type)
    embeddings = embed_texts([c.text for c in chunks])
    embedded_chunks = [replace(c, embedding=e) for c, e in zip(chunks, embeddings)]

    vector_store.add_chunks(session_id, embedded_chunks)
    result.chunk_count = len(chunks)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_indexing.py -v`
Expected: all 4 PASS.

Note: `pytest tests/` (the full suite) will still show failures in `tests/test_retriever.py` and `tests/test_answer.py` at this point — both call `index_document(...)` without `session_id` in their own setup code. That's expected; Task 3 and Task 4 fix those files.

- [ ] **Step 5: Commit**

```bash
git add core/indexing.py tests/test_indexing.py
git commit -m "feat: thread session_id through index_document"
```

---

### Task 3: Thread session_id through retrieve

**Files:**
- Modify: `core/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `VectorStore.query(session_id, document_id, query_embedding, k)`, `VectorStore.get_all_chunks(session_id, document_id)` (Task 1); `index_document(file_path, document_id, vector_store, session_id)` (Task 2).
- Produces: `retrieve(vector_store, session_id, document_ids, question) -> list[dict]` — used by Task 4 (`answer_question`) and Task 9 (reranking, which replaces this task's factual branch body).

- [ ] **Step 1: Update the failing tests**

Replace the entire contents of `tests/test_retriever.py` with:

```python
from pathlib import Path

from core.indexing import index_document
from core.retriever import dedupe_by_id, detect_question_type, extract_keywords, keyword_search, retrieve
from core.vectorstore import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
SESSION = "test-session"


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
        session_id=SESSION,
    )

    results = retrieve(store, SESSION, ["pdf-doc"], "What is DE Monitoring?")

    assert 0 < len(results) <= 5
    assert any(r["location_label"] == "p.4" for r in results)
    assert all(r["document_id"] == "pdf-doc" for r in results)


def test_retrieve_exhaustive_question_finds_keyword_matches_semantic_search_could_miss(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(
        str(FIXTURES / "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf"),
        document_id="pdf-doc",
        vector_store=store,
        session_id=SESSION,
    )

    results = retrieve(store, SESSION, ["pdf-doc"], "Find all references to prompt injection")

    assert any(r["location_label"] == "p.22" for r in results)
    assert len(results) > 5  # broader than a plain factual top-k


def test_retrieve_merges_factual_results_across_documents_and_caps_globally(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("The quarterly revenue report shows steady growth.\n" * 5)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("The lease termination clause requires 30 days notice.\n" * 5)
    index_document(str(doc_a), document_id="doc-a", vector_store=store, session_id=SESSION)
    index_document(str(doc_b), document_id="doc-b", vector_store=store, session_id=SESSION)

    results = retrieve(store, SESSION, ["doc-a", "doc-b"], "What does the lease termination clause require?")

    assert 0 < len(results) <= 5
    assert any(r["document_id"] == "doc-b" for r in results)


def test_retrieve_exhaustive_merges_keyword_matches_across_documents(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("Penalty clause: late delivery incurs a penalty fee.\n" * 3)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("Separate penalty clause: early termination incurs a penalty fee.\n" * 3)
    index_document(str(doc_a), document_id="doc-a", vector_store=store, session_id=SESSION)
    index_document(str(doc_b), document_id="doc-b", vector_store=store, session_id=SESSION)

    results = retrieve(store, SESSION, ["doc-a", "doc-b"], "Find all references to penalty")

    found_document_ids = {r["document_id"] for r in results}
    assert found_document_ids == {"doc-a", "doc-b"}


def test_retrieve_isolates_results_by_session_even_for_the_same_document_id(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("Session A: the launch code is ALPHA-7.\n" * 3)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("Session B: the launch code is BRAVO-9.\n" * 3)
    index_document(str(doc_a), document_id="shared-doc", vector_store=store, session_id="session-a")
    index_document(str(doc_b), document_id="shared-doc", vector_store=store, session_id="session-b")

    results_a = retrieve(store, "session-a", ["shared-doc"], "What is the launch code?")
    results_b = retrieve(store, "session-b", ["shared-doc"], "What is the launch code?")

    assert any("ALPHA-7" in r["text"] for r in results_a)
    assert all("BRAVO-9" not in r["text"] for r in results_a)
    assert any("BRAVO-9" in r["text"] for r in results_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL — `TypeError: retrieve() missing 1 required positional argument` (the calls now pass `session_id` but `retrieve()` doesn't accept it yet).

- [ ] **Step 3: Implement**

Replace the entire contents of `core/retriever.py` with:

```python
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


def retrieve(vector_store: VectorStore, session_id: str, document_ids: list[str], question: str) -> list[dict]:
    question_type = detect_question_type(question)
    query_embedding = embed_texts([question])[0]

    if question_type == "factual":
        merged = []
        for document_id in document_ids:
            for match in vector_store.query(session_id, document_id, query_embedding, k=FACTUAL_K):
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
        for match in vector_store.query(session_id, document_id, query_embedding, k=BROAD_K):
            match["document_id"] = document_id
            semantic_results.append(match)
        doc_chunks = vector_store.get_all_chunks(session_id, document_id)
        for chunk in doc_chunks:
            chunk["document_id"] = document_id
        keyword_results.extend(keyword_search(doc_chunks, keywords))

    semantic_results.sort(key=lambda m: m["distance"])
    return dedupe_by_id(keyword_results + semantic_results[:BROAD_K])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: all PASS (12 tests — 11 existing + 1 new).

Note: `tests/test_answer.py` still fails at this point (calls `index_document`/would-call `retrieve` without `session_id`) — expected, fixed in Task 4.

- [ ] **Step 5: Commit**

```bash
git add core/retriever.py tests/test_retriever.py
git commit -m "feat: thread session_id through retrieve"
```

---

### Task 4: Thread session_id through answer_question and build_answer_message

**Files:**
- Modify: `core/answer.py`, `core/chat_handler.py`
- Test: `tests/test_answer.py`, `tests/test_chat_handler.py`

**Interfaces:**
- Consumes: `retrieve(vector_store, session_id, document_ids, question)` (Task 3).
- Produces: `answer_question(vector_store, session_id, document_ids, question, document_names=None, llm=None, chat_history=None) -> Answer`; `build_answer_message(vector_store, session_id, document_ids, question, document_names=None, chat_history=None, answer_fn=answer_question) -> dict`. Used by `app.py` (Task 5) and `eval/run_eval.py` (Task 11).

- [ ] **Step 1: Update the failing tests**

Replace the entire contents of `tests/test_answer.py` with:

```python
from pathlib import Path

from core.answer import answer_question
from core.indexing import index_document
from core.prompt import NOT_FOUND_MESSAGE
from core.vectorstore import VectorStore

FIXTURES = Path(__file__).parent / "fixtures"
SESSION = "test-session"


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response


def _indexed_store(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    index_document(str(FIXTURES / "sample.txt"), document_id="txt-doc", vector_store=store, session_id=SESSION)
    return store


def test_answer_question_returns_grounded_answer_with_sources(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM("The fox jumps over the lazy dog. [lines 1-20]")

    answer = answer_question(store, SESSION, ["txt-doc"], "What does the fox do?", llm=llm)

    assert answer.text == "The fox jumps over the lazy dog. [lines 1-20]"
    assert answer.found_in_document is True
    assert len(answer.sources) > 0
    assert all(s.location_label.startswith("lines ") for s in answer.sources)
    assert all(s.document_id == "txt-doc" for s in answer.sources)


def test_answer_question_detects_not_found_response(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM(NOT_FOUND_MESSAGE)

    answer = answer_question(store, SESSION, ["txt-doc"], "What is the capital of France?", llm=llm)

    assert answer.found_in_document is False
    assert answer.sources == []


def test_answer_question_passes_chat_history_into_the_prompt(tmp_path):
    store = _indexed_store(tmp_path)
    llm = _FakeLLM("some answer")
    history = [("user", "earlier question"), ("assistant", "earlier answer")]

    answer_question(store, SESSION, ["txt-doc"], "a follow-up question", llm=llm, chat_history=history)

    assert "earlier question" in llm.last_prompt
    assert "earlier answer" in llm.last_prompt


def test_answer_question_citations_span_multiple_documents(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("The quarterly revenue report shows steady growth.\n" * 3)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("The lease termination clause requires 30 days notice.\n" * 3)
    index_document(str(doc_a), document_id="doc-a", vector_store=store, session_id=SESSION)
    index_document(str(doc_b), document_id="doc-b", vector_store=store, session_id=SESSION)
    llm = _FakeLLM("The lease requires 30 days notice. [Lease.txt — lines 1-3]")

    answer = answer_question(
        store,
        SESSION,
        ["doc-a", "doc-b"],
        "What does the lease termination clause require?",
        document_names={"doc-a": "Revenue.txt", "doc-b": "Lease.txt"},
        llm=llm,
    )

    assert any(s.document_id == "doc-b" for s in answer.sources)
    assert "Lease.txt" in llm.last_prompt
    assert "Revenue.txt" in llm.last_prompt


def test_answer_question_isolates_sessions_even_for_the_same_document_id(tmp_path):
    store = VectorStore(persist_directory=str(tmp_path))
    doc_a = tmp_path / "doc_a.txt"
    doc_a.write_text("Session A: the launch code is ALPHA-7.\n" * 3)
    doc_b = tmp_path / "doc_b.txt"
    doc_b.write_text("Session B: the launch code is BRAVO-9.\n" * 3)
    index_document(str(doc_a), document_id="shared-doc", vector_store=store, session_id="session-a")
    index_document(str(doc_b), document_id="shared-doc", vector_store=store, session_id="session-b")
    llm = _FakeLLM("some answer")

    answer_question(store, "session-a", ["shared-doc"], "What is the launch code?", llm=llm)

    assert "ALPHA-7" in llm.last_prompt
    assert "BRAVO-9" not in llm.last_prompt
```

Replace the entire contents of `tests/test_chat_handler.py` with:

```python
import httpx
from groq import RateLimitError

from core.chat_handler import build_answer_message
from core.models import Answer, Source

SESSION = "test-session"


def _rate_limit_error():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com/x"))
    return RateLimitError("rate limited", response=response, body=None)


def test_successful_answer_produces_assistant_message_with_named_sources():
    def fake_answer_fn(vector_store, session_id, document_ids, question, document_names=None, chat_history=None):
        return Answer(
            text="The due date is the 1st.",
            found_in_document=True,
            sources=[Source(location_label="p.1", snippet="Payment due date is the 1st.", document_id="doc1")],
        )

    message = build_answer_message(
        None, SESSION, ["doc1"], "When is it due?", document_names={"doc1": "Contract.pdf"}, answer_fn=fake_answer_fn
    )

    assert message == {
        "role": "assistant",
        "text": "The due date is the 1st.",
        "sources": ["Contract.pdf — p.1"],
        "found_in_document": True,
    }


def test_source_without_a_known_document_name_falls_back_to_bare_location_label():
    def fake_answer_fn(vector_store, session_id, document_ids, question, document_names=None, chat_history=None):
        return Answer(
            text="The due date is the 1st.",
            found_in_document=True,
            sources=[Source(location_label="p.1", snippet="...", document_id="doc1")],
        )

    message = build_answer_message(None, SESSION, ["doc1"], "When is it due?", answer_fn=fake_answer_fn)

    assert message["sources"] == ["p.1"]


def test_not_found_answer_produces_message_with_no_sources():
    def fake_answer_fn(vector_store, session_id, document_ids, question, document_names=None, chat_history=None):
        return Answer(
            text="I couldn't find that in the document.", found_in_document=False, sources=[]
        )

    message = build_answer_message(None, SESSION, ["doc1"], "?", answer_fn=fake_answer_fn)

    assert message["found_in_document"] is False
    assert message["sources"] == []


def test_rate_limit_error_produces_friendly_retry_message():
    def failing_answer_fn(vector_store, session_id, document_ids, question, document_names=None, chat_history=None):
        raise _rate_limit_error()

    message = build_answer_message(None, SESSION, ["doc1"], "?", answer_fn=failing_answer_fn)

    assert message["role"] == "assistant"
    assert message["found_in_document"] is False
    assert message["sources"] == []
    assert "rate limit" in message["text"].lower()


def test_generic_error_produces_distinct_message_not_mentioning_rate_limit():
    def failing_answer_fn(vector_store, session_id, document_ids, question, document_names=None, chat_history=None):
        raise RuntimeError("boom")

    message = build_answer_message(None, SESSION, ["doc1"], "?", answer_fn=failing_answer_fn)

    assert message["found_in_document"] is False
    assert "rate limit" not in message["text"].lower()
    assert "went wrong" in message["text"].lower()


def test_generic_error_logs_the_real_exception_for_diagnosis(caplog):
    def failing_answer_fn(vector_store, session_id, document_ids, question, document_names=None, chat_history=None):
        raise RuntimeError("boom")

    with caplog.at_level("ERROR"):
        build_answer_message(None, SESSION, ["doc1"], "?", answer_fn=failing_answer_fn)

    assert "boom" in caplog.text


def test_answer_fn_is_called_with_vector_store_session_document_ids_names_and_history():
    received = {}

    def fake_answer_fn(vector_store, session_id, document_ids, question, document_names=None, chat_history=None):
        received["args"] = (vector_store, session_id, document_ids, question, document_names, chat_history)
        return Answer(text="ok", found_in_document=True, sources=[])

    sentinel_store = object()
    build_answer_message(
        sentinel_store,
        SESSION,
        ["doc1"],
        "What?",
        document_names={"doc1": "Contract.pdf"},
        chat_history=[("user", "prior")],
        answer_fn=fake_answer_fn,
    )

    assert received["args"] == (
        sentinel_store,
        SESSION,
        ["doc1"],
        "What?",
        {"doc1": "Contract.pdf"},
        [("user", "prior")],
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_answer.py tests/test_chat_handler.py -v`
Expected: FAIL — `TypeError: answer_question() missing 1 required positional argument` and `TypeError: fake_answer_fn() takes from 4 to 6 positional arguments but 5 were given` (or similar), since neither function accepts `session_id` yet.

- [ ] **Step 3: Implement**

Replace the entire contents of `core/answer.py` with:

```python
from core.llm import GroqLLM
from core.models import Answer, Source
from core.prompt import NOT_FOUND_MESSAGE, build_prompt
from core.retriever import retrieve
from core.vectorstore import VectorStore

SNIPPET_LENGTH = 200


def answer_question(
    vector_store: VectorStore,
    session_id: str,
    document_ids: list[str],
    question: str,
    document_names: dict[str, str] | None = None,
    llm=None,
    chat_history: list[tuple[str, str]] | None = None,
) -> Answer:
    document_names = document_names or {}
    chunks = retrieve(vector_store, session_id, document_ids, question)
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

Replace the entire contents of `core/chat_handler.py` with:

```python
import logging

from groq import RateLimitError

from core.answer import answer_question

logger = logging.getLogger(__name__)

RATE_LIMIT_MESSAGE = (
    "Hit the free-tier rate limit, retrying in a few seconds… please try again shortly."
)
GENERIC_ERROR_MESSAGE = "Something went wrong answering that question. Please try again."


def _format_source(source, document_names: dict[str, str]) -> str:
    name = document_names.get(source.document_id)
    return f"{name} — {source.location_label}" if name else source.location_label


def build_answer_message(
    vector_store,
    session_id: str,
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
            session_id,
            document_ids,
            question,
            document_names=document_names,
            chat_history=chat_history,
        )
    except RateLimitError:
        return {"role": "assistant", "text": RATE_LIMIT_MESSAGE, "sources": [], "found_in_document": False}
    except Exception:
        logger.exception("Answering question failed: %r", question)
        return {"role": "assistant", "text": GENERIC_ERROR_MESSAGE, "sources": [], "found_in_document": False}

    return {
        "role": "assistant",
        "text": answer.text,
        "sources": [_format_source(s, document_names) for s in answer.sources],
        "found_in_document": answer.found_in_document,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_answer.py tests/test_chat_handler.py -v`
Expected: all PASS (5 in `test_answer.py` — 4 existing + 1 new; 7 in `test_chat_handler.py`, all existing, updated signatures).

- [ ] **Step 5: Run the full suite to confirm session isolation is fully wired**

Run: `pytest tests/ -v`
Expected: all PASS. `app.py` itself hasn't been updated yet (Task 5), so the app won't run correctly until then, but every `core/` module and test now agrees on the `session_id`-threaded signatures.

- [ ] **Step 6: Commit**

```bash
git add core/answer.py core/chat_handler.py tests/test_answer.py tests/test_chat_handler.py
git commit -m "feat: thread session_id through answer_question and build_answer_message"
```

---

### Task 5: Generate and wire session_id in app.py

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `index_document(file_path, document_id, vector_store, session_id)` (Task 2), `VectorStore.delete_document(session_id, document_id)` (Task 1), `build_answer_message(vector_store, session_id, document_ids, question, ...)` (Task 4).
- Produces: nothing new for later tasks — this is the final integration point. No automated test file, consistent with the rest of this project's `app.py` (thin UI glue, verified manually/via browser).

- [ ] **Step 1: Add session_id generation**

In `app.py`, add `import uuid` to the top-level imports (alongside `hashlib`, `tempfile`, `pathlib`):

```python
import hashlib
import tempfile
import uuid
from pathlib import Path
```

Then, right after the existing `if "documents" not in st.session_state:` block, add a new block:

```python
if "documents" not in st.session_state:
    st.session_state.documents = {}
    reset_chat()
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:16]
if "removed_document_ids" not in st.session_state:
    st.session_state.removed_document_ids = set()
```

- [ ] **Step 2: Pass session_id into index_document**

Find:

```python
        try:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                result = index_document(
                    tmp_path, document_id=document_id, vector_store=get_vector_store()
                )
```

Replace with:

```python
        try:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                result = index_document(
                    tmp_path,
                    document_id=document_id,
                    vector_store=get_vector_store(),
                    session_id=st.session_state.session_id,
                )
```

- [ ] **Step 3: Pass session_id into delete_document**

Find:

```python
            if remove_clicked:
                get_vector_store().delete_document(document_id)
```

Replace with:

```python
            if remove_clicked:
                get_vector_store().delete_document(st.session_state.session_id, document_id)
```

- [ ] **Step 4: Pass session_id into both build_answer_message call sites**

Find (inside the summarize-picker branch):

```python
                with st.spinner("Summarizing…"):
                    message = build_answer_message(
                        get_vector_store(),
                        [chosen_id],
                        question,
                        document_names=document_names,
                        chat_history=history_for_prompt,
                    )
```

Replace with:

```python
                with st.spinner("Summarizing…"):
                    message = build_answer_message(
                        get_vector_store(),
                        st.session_state.session_id,
                        [chosen_id],
                        question,
                        document_names=document_names,
                        chat_history=history_for_prompt,
                    )
```

Find (in the direct-answer branch):

```python
            with st.spinner("Searching documents…"):
                message = build_answer_message(
                    get_vector_store(),
                    document_ids,
                    question,
                    document_names=document_names,
                    chat_history=history_for_prompt,
                )
```

Replace with:

```python
            with st.spinner("Searching documents…"):
                message = build_answer_message(
                    get_vector_store(),
                    st.session_state.session_id,
                    document_ids,
                    question,
                    document_names=document_names,
                    chat_history=history_for_prompt,
                )
```

- [ ] **Step 5: Run the full automated suite**

Run: `pytest tests/ -v`
Expected: all PASS (no `core/` signatures changed in this task, only `app.py`).

- [ ] **Step 6: Manually verify session isolation**

Run: `streamlit run app.py`

Open the app in two different browsers (or one normal window + one incognito/private window, so each gets a separate session cookie — a single browser's two tabs share the same Streamlit session). In each:
1. Upload a different document per browser.
2. Confirm each browser only ever sees its own document(s) in its "Documents (N/10)" list — never the other browser's.
3. In one browser, remove its document via ✕. Confirm the other browser's document is unaffected (still indexed, still answerable).

Record the outcome in your task notes — this task has no automated test file for `app.py`, consistent with the rest of the project.

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: generate a per-session session_id and wire it through app.py"
```

---

### Task 6: core/ocr.py — Tesseract fallback for image-only pages

**Files:**
- Create: `core/ocr.py`
- Test: `tests/test_ocr.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `pymupdf.Page` (an already-open page object), the `pytesseract` package, the system `tesseract` binary (see Global Constraints prerequisite).
- Produces: `ocr_page_text(page: pymupdf.Page) -> str` — returns the OCR'd text, or `""` if OCR fails for any reason. Used by Task 7 (`core/ingest.py`).

- [ ] **Step 1: Add the new dependency**

In `requirements.txt`, add a line:

```
pytesseract
```

Install it: `pip install pytesseract` (prefix with `CURL_CA_BUNDLE=` on this machine if pip needs it, per existing project convention).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_ocr.py`:

```python
import pymupdf
import pytesseract

from core.ocr import ocr_page_text


def _image_only_pdf_page(tmp_path, text):
    # Render real vector text at high resolution, then embed the RENDERED
    # IMAGE (not the text) as the only content of a fresh page, so
    # page.get_text() finds nothing and OCR is the only way to recover it.
    text_pdf = pymupdf.open()
    text_page = text_pdf.new_page()
    text_page.insert_text((72, 200), text, fontsize=36)
    pix = text_page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
    image_path = tmp_path / "rendered.png"
    pix.save(str(image_path))
    text_pdf.close()

    image_pdf = pymupdf.open()
    image_page = image_pdf.new_page()
    image_page.insert_image(image_page.rect, filename=str(image_path))
    pdf_path = tmp_path / "image_only.pdf"
    image_pdf.save(str(pdf_path))
    image_pdf.close()
    return pdf_path


def test_ocr_page_text_extracts_text_from_an_image_only_page(tmp_path):
    pdf_path = _image_only_pdf_page(tmp_path, "HELLO OCR WORLD")

    with pymupdf.open(str(pdf_path)) as pdf:
        page = pdf[0]
        assert page.get_text().strip() == ""  # confirm there is no real text layer

        result = ocr_page_text(page)

    assert "HELLO" in result.upper()


def test_ocr_page_text_returns_empty_string_when_ocr_fails(tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", _boom)
    pdf_path = _image_only_pdf_page(tmp_path, "irrelevant")

    with pymupdf.open(str(pdf_path)) as pdf:
        result = ocr_page_text(pdf[0])

    assert result == ""
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.ocr'`.

- [ ] **Step 4: Implement**

Create `core/ocr.py`:

```python
import logging
import os
import tempfile

import pymupdf
import pytesseract

logger = logging.getLogger(__name__)

if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]


def ocr_page_text(page: pymupdf.Page) -> str:
    try:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pix.save(tmp_path)
            return pytesseract.image_to_string(tmp_path).strip()
        finally:
            os.remove(tmp_path)
    except Exception:
        logger.exception("OCR failed for a page; treating it as having no extractable text")
        return ""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ocr.py -v`
Expected: both PASS. (If the first test fails with a `TesseractNotFoundError` traceback instead, Tesseract isn't installed/on `PATH` — see the Global Constraints prerequisite section at the top of this plan.)

- [ ] **Step 6: Commit**

```bash
git add core/ocr.py tests/test_ocr.py requirements.txt
git commit -m "feat: add Tesseract OCR fallback for image-only PDF pages"
```

---

### Task 7: Integrate OCR into PDF ingestion + deployment config

**Files:**
- Modify: `core/ingest.py`
- Test: `tests/test_ingest.py`
- Create: `packages.txt`
- Modify: `run_steps.txt`, `README.md`

**Interfaces:**
- Consumes: `ocr_page_text(page) -> str` (Task 6).
- Produces: `ingest_document(file_path) -> IngestResult` now transcribes image-only PDF pages instead of skipping them. No signature change — same interface the rest of the pipeline already depends on.

- [ ] **Step 1: Write the failing test**

In `tests/test_ingest.py`, add this helper right after the existing `_build_pdf` helper, and this test at the end of the file:

```python
def _build_image_only_pdf(path: Path, text: str):
    """A PDF page containing only a rendered image of `text`, no text layer —
    simulates a scanned page so OCR is the only way to recover the text."""
    text_pdf = pymupdf.open()
    text_page = text_pdf.new_page()
    text_page.insert_text((72, 200), text, fontsize=36)
    pix = text_page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
    image_path = path.parent / f"{path.stem}_rendered.png"
    pix.save(str(image_path))
    text_pdf.close()

    image_pdf = pymupdf.open()
    image_page = image_pdf.new_page()
    image_page.insert_image(image_page.rect, filename=str(image_path))
    image_pdf.save(str(path))
    image_pdf.close()
```

```python
def test_load_pdf_uses_ocr_when_a_page_has_no_text_layer(tmp_path):
    pdf_path = tmp_path / "scanned_but_readable.pdf"
    _build_image_only_pdf(pdf_path, "OCR FALLBACK WORKS")

    result = ingest_document(str(pdf_path))

    assert result.source_type == "pdf"
    assert result.has_extractable_text is True
    assert any("OCR" in seg.text.upper() for seg in result.segments)
    assert result.segments[0].location_label == "p.1"
```

- [ ] **Step 2: Run tests to verify the new test fails**

Run: `pytest tests/test_ingest.py -v`
Expected: `test_load_pdf_uses_ocr_when_a_page_has_no_text_layer` FAILS — `assert False is True` on `has_extractable_text` (the page is currently just skipped, so `result.segments == []`). All other existing tests still PASS.

- [ ] **Step 3: Implement**

In `core/ingest.py`, add the import at the top:

```python
import os

import docx
import docx.opc.exceptions
import pymupdf

from core.models import ExtractedSegment
from core.ocr import ocr_page_text
```

In `_load_pdf`, find:

```python
            if not text:
                continue
            segments.append(
                ExtractedSegment(text=text, location_label=f"p.{page_number}")
            )
```

Replace with:

```python
            if not text:
                text = ocr_page_text(page)
            if not text:
                continue
            segments.append(
                ExtractedSegment(text=text, location_label=f"p.{page_number}")
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ingest.py -v`
Expected: all PASS (14 tests — 13 existing + 1 new).

- [ ] **Step 5: Add Streamlit Cloud deployment config**

Create `packages.txt` at the project root (this is the file Streamlit Community Cloud reads to install system/apt packages before Python dependencies):

```
tesseract-ocr
```

- [ ] **Step 6: Update setup docs**

In `run_steps.txt`, in the "3. Install dependencies" section, add a new step after the existing torch note:

```
   Scanned/image-only PDFs are transcribed via Tesseract OCR automatically.
   Install the Tesseract binary locally (not a pip package):
     Windows: https://github.com/UB-Mannheim/tesseract/wiki
       If `tesseract` isn't on PATH afterward, set an environment variable
       TESSERACT_CMD to the full path of tesseract.exe.
     macOS:   brew install tesseract
     Linux:   sudo apt-get install tesseract-ocr
   (Streamlit Cloud installs this automatically via packages.txt — no action
   needed for that deployment target.)
```

In `README.md`'s "Known limitations (MVP)" section, replace the line:

```
- No OCR — scanned/image-only PDFs will show a warning, not extracted text
```

with:

```
- Scanned/image-only PDFs are OCR'd automatically via Tesseract; if OCR still
  finds nothing (e.g. a genuinely blank page), the existing "looks scanned"
  warning is shown instead of extracted text
```

- [ ] **Step 7: Run the full suite**

Run: `pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add core/ingest.py tests/test_ingest.py packages.txt run_steps.txt README.md
git commit -m "feat: OCR image-only PDF pages instead of skipping them"
```

---

### Task 8: core/reranker.py — cross-encoder reranking

**Files:**
- Create: `core/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `sentence_transformers.CrossEncoder` (already a transitive part of the `sentence-transformers` dependency already in `requirements.txt` — no new dependency needed).
- Produces: `rerank(question: str, candidates: list[dict]) -> list[dict]` — returns `candidates` reordered by relevance to `question`, most relevant first. Used by Task 9 (`core/retriever.py`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reranker.py`:

```python
from core.reranker import rerank


def test_rerank_orders_the_more_relevant_candidate_first():
    candidates = [
        {"id": "a", "text": "The sky is blue and birds fly in it."},
        {"id": "b", "text": "The termination clause requires 30 days written notice."},
    ]

    results = rerank("What is the termination notice period?", candidates)

    assert results[0]["id"] == "b"


def test_rerank_returns_empty_list_for_no_candidates():
    assert rerank("anything", []) == []


def test_rerank_preserves_all_candidates_just_reordered():
    candidates = [
        {"id": "a", "text": "Irrelevant filler text about gardening."},
        {"id": "b", "text": "More irrelevant filler about cooking."},
        {"id": "c", "text": "The payment due date is the first of every month."},
    ]

    results = rerank("When is the payment due?", candidates)

    assert {r["id"] for r in results} == {"a", "b", "c"}
    assert results[0]["id"] == "c"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reranker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.reranker'`.

- [ ] **Step 3: Implement**

Create `core/reranker.py`:

```python
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    model = _get_model()
    pairs = [[question, c["text"]] for c in candidates]
    scores = model.predict(pairs)
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reranker.py -v`
Expected: all 3 PASS. (First run downloads the `cross-encoder/ms-marco-MiniLM-L-6-v2` model from Hugging Face — a few seconds to a minute depending on connection; cached afterward, same pattern as the existing `all-MiniLM-L6-v2` embedding model.)

- [ ] **Step 5: Commit**

```bash
git add core/reranker.py tests/test_reranker.py
git commit -m "feat: add cross-encoder reranker"
```

---

### Task 9: Integrate reranking into factual retrieval

**Files:**
- Modify: `core/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `rerank(question, candidates) -> list[dict]` (Task 8).
- Produces: `retrieve(...)`'s factual branch now reranks a widened candidate pool instead of sorting by raw embedding distance. Signature unchanged from Task 3 — `answer_question` (Task 4) needs no changes.

- [ ] **Step 1: Write the failing test**

In `tests/test_retriever.py`, add this import to the top (alongside the existing ones) and this test at the end of the file:

```python
from core.retriever import dedupe_by_id, detect_question_type, extract_keywords, keyword_search, retrieve
```
(this import line already exists — just confirm `retrieve` is present; no change needed to the import itself)

```python
def test_retrieve_factual_path_delegates_final_ranking_to_rerank(tmp_path, monkeypatch):
    store = VectorStore(persist_directory=str(tmp_path))
    doc = tmp_path / "doc.txt"
    doc.write_text("Filler sentence about something unrelated.\n" * 5)
    index_document(str(doc), document_id="doc-1", vector_store=store, session_id=SESSION)

    sentinel = [{"id": "sentinel-1", "text": "sentinel"}, {"id": "sentinel-2", "text": "sentinel"}]

    def fake_rerank(question, candidates):
        assert len(candidates) > 0
        return sentinel

    monkeypatch.setattr("core.retriever.rerank", fake_rerank)

    results = retrieve(store, SESSION, ["doc-1"], "What is this about?")

    assert results == sentinel
```

- [ ] **Step 2: Run tests to verify the new test fails**

Run: `pytest tests/test_retriever.py -v`
Expected: `test_retrieve_factual_path_delegates_final_ranking_to_rerank` FAILS — either `AttributeError: <module 'core.retriever'> does not have the attribute 'rerank'` (nothing to monkeypatch yet) or an assertion mismatch, since the factual branch doesn't call `rerank` yet. All other tests still PASS.

- [ ] **Step 3: Implement**

In `core/retriever.py`, add the import and new constant:

```python
import re

from core.embedder import embed_texts
from core.reranker import rerank
from core.vectorstore import VectorStore

FACTUAL_K = 5
FACTUAL_CANDIDATE_K = 12
BROAD_K = 12
```

Find the factual branch inside `retrieve`:

```python
    if question_type == "factual":
        merged = []
        for document_id in document_ids:
            for match in vector_store.query(session_id, document_id, query_embedding, k=FACTUAL_K):
                match["document_id"] = document_id
                merged.append(match)
        merged.sort(key=lambda m: m["distance"])
        return dedupe_by_id(merged)[:FACTUAL_K]
```

Replace with:

```python
    if question_type == "factual":
        merged = []
        for document_id in document_ids:
            for match in vector_store.query(session_id, document_id, query_embedding, k=FACTUAL_CANDIDATE_K):
                match["document_id"] = document_id
                merged.append(match)
        candidates = dedupe_by_id(merged)
        return rerank(question, candidates)[:FACTUAL_K]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: all PASS (13 tests — 12 from Task 3 + 1 new). The 3 pre-existing factual-path tests (`test_retrieve_factual_question_returns_a_small_top_k`, `test_retrieve_merges_factual_results_across_documents_and_caps_globally`, `test_retrieve_isolates_results_by_session_even_for_the_same_document_id`) need no changes — they assert on real document content where the correct chunk is unambiguously the most relevant, so reranking doesn't change the outcome, only how it's reached.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add core/retriever.py tests/test_retriever.py
git commit -m "feat: rerank factual retrieval candidates with a cross-encoder"
```

---

### Task 10: Curated evaluation dataset + pure scoring functions

**Files:**
- Create: `eval/qa_dataset.json`, `eval/scoring.py`
- Test: `tests/test_eval_scoring.py`

**Interfaces:**
- Consumes: `core.models.Source` (existing).
- Produces: `score_retrieval_hit(sources: list[Source], expected_locations: list[str]) -> bool`, `score_answer_coverage(answer_text: str, expected_keywords: list[str]) -> bool`, `write_report(results: list[dict], path: str) -> None`. Used by Task 11 (`eval/run_eval.py`).

- [ ] **Step 1: Write the curated dataset**

Create `eval/qa_dataset.json`. Every entry is grounded in the real content of `tests/fixtures/29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf` (24 pages) and `tests/fixtures/Build a RAG Application with LlamaIndex.docx`:

```json
[
  {
    "id": "q1",
    "question": "What is DE Monitoring?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["operational"],
    "expected_locations": ["p.4"]
  },
  {
    "id": "q2",
    "question": "What is Agent Monitoring?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["hallucinating"],
    "expected_locations": ["p.4"]
  },
  {
    "id": "q3",
    "question": "What does the LANGCHAIN_TRACING_V2 environment variable do?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["tracing"],
    "expected_locations": ["p.12"]
  },
  {
    "id": "q4",
    "question": "What technique does the retry_with_backoff decorator implement?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["exponential backoff"],
    "expected_locations": ["p.18"]
  },
  {
    "id": "q5",
    "question": "What is the biggest financial security threat in a GenAI system?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["API key"],
    "expected_locations": ["p.21"]
  },
  {
    "id": "q6",
    "question": "What is a prompt injection attack?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["instructions"],
    "expected_locations": ["p.22"]
  },
  {
    "id": "q7",
    "question": "How should sensitive data like emails be handled in logs?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["mask"],
    "expected_locations": ["p.23"]
  },
  {
    "id": "q8",
    "question": "What does rate limiting protect a GenAI application against?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["Denial-of-Service"],
    "expected_locations": ["p.23"]
  },
  {
    "id": "q9",
    "question": "What can someone do if they obtain your API key?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["billing"],
    "expected_locations": ["p.21"]
  },
  {
    "id": "q10",
    "question": "What is the purpose of guardrails in error handling?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["validation"],
    "expected_locations": ["p.17"]
  },
  {
    "id": "q11",
    "question": "What should you add to .gitignore to protect your API key?",
    "type": "factual",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": [".env"],
    "expected_locations": ["p.21"]
  },
  {
    "id": "q12",
    "question": "List all three types of failures in GenAI systems.",
    "type": "exhaustive",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["Network Failures", "Output Failures", "Agent Failures"],
    "expected_locations": ["p.17"]
  },
  {
    "id": "q13",
    "question": "Find all references to prompt injection in this document.",
    "type": "exhaustive",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["injection"],
    "expected_locations": ["p.22"]
  },
  {
    "id": "q14",
    "question": "What are the four guardrails every production GenAI system must have?",
    "type": "exhaustive",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["API Key Protection", "Rate Limiting"],
    "expected_locations": ["p.23", "p.24"]
  },
  {
    "id": "q15",
    "question": "Summarize how this document says GenAI monitoring differs from traditional monitoring.",
    "type": "summarization",
    "document": "29_Summary_ Monitoring & Securing GenAI Systems-6275.pdf",
    "expected_keywords": ["hallucin"],
    "expected_locations": []
  },
  {
    "id": "q16",
    "question": "Which embedding model does this guide use?",
    "type": "factual",
    "document": "Build a RAG Application with LlamaIndex.docx",
    "expected_keywords": ["gemini-embedding-001"],
    "expected_locations": ["para 7 (Document)"]
  },
  {
    "id": "q17",
    "question": "Which LLM model does this guide use via Ollama Cloud?",
    "type": "factual",
    "document": "Build a RAG Application with LlamaIndex.docx",
    "expected_keywords": ["gemma4:31b-cloud"],
    "expected_locations": ["para 10 (Document)"]
  },
  {
    "id": "q18",
    "question": "How many top chunks does the interactive query loop retrieve?",
    "type": "factual",
    "document": "Build a RAG Application with LlamaIndex.docx",
    "expected_keywords": ["top 3"],
    "expected_locations": ["para 16 (Document)"]
  },
  {
    "id": "q19",
    "question": "What must be included in requirements.txt according to this guide?",
    "type": "factual",
    "document": "Build a RAG Application with LlamaIndex.docx",
    "expected_keywords": ["pinned dependencies"],
    "expected_locations": ["para 66 (Document)"]
  },
  {
    "id": "q20",
    "question": "What are the hard rules listed in this guide?",
    "type": "exhaustive",
    "document": "Build a RAG Application with LlamaIndex.docx",
    "expected_keywords": ["hardcoded API keys", "streaming"],
    "expected_locations": ["para 57 (Document)", "para 58 (Document)"]
  }
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_eval_scoring.py`:

```python
from core.models import Source
from eval.scoring import score_answer_coverage, score_retrieval_hit, write_report


def test_score_retrieval_hit_true_when_any_expected_location_present():
    sources = [
        Source(location_label="p.3", snippet="", document_id="d"),
        Source(location_label="p.4", snippet="", document_id="d"),
    ]
    assert score_retrieval_hit(sources, ["p.4", "p.9"]) is True


def test_score_retrieval_hit_false_when_no_expected_location_present():
    sources = [Source(location_label="p.3", snippet="", document_id="d")]
    assert score_retrieval_hit(sources, ["p.4"]) is False


def test_score_retrieval_hit_true_when_no_expected_locations_given():
    assert score_retrieval_hit([], []) is True


def test_score_answer_coverage_true_when_all_keywords_present_case_insensitive():
    assert score_answer_coverage("The API Key must be PROTECTED.", ["api key", "protected"]) is True


def test_score_answer_coverage_false_when_a_keyword_is_missing():
    assert score_answer_coverage("The API Key must be protected.", ["api key", "encryption"]) is False


def test_score_answer_coverage_true_when_no_keywords_given():
    assert score_answer_coverage("anything", []) is True


def test_write_report_creates_markdown_with_summary_and_table(tmp_path):
    results = [
        {"id": "q1", "question": "Q1?", "type": "factual", "retrieval_hit": True, "answer_hit": True},
        {"id": "q2", "question": "Q2?", "type": "factual", "retrieval_hit": False, "answer_hit": True},
    ]
    report_path = tmp_path / "report.md"

    write_report(results, str(report_path))

    content = report_path.read_text(encoding="utf-8")
    assert "Retrieval hit-rate: 1/2" in content
    assert "Overall pass (both): 1/2" in content
    assert "q1" in content and "q2" in content
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_eval_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.scoring'`.

- [ ] **Step 4: Implement**

Create `eval/scoring.py`:

```python
def score_retrieval_hit(sources, expected_locations: list[str]) -> bool:
    if not expected_locations:
        return True
    retrieved_labels = {s.location_label for s in sources}
    return any(loc in retrieved_labels for loc in expected_locations)


def score_answer_coverage(answer_text: str, expected_keywords: list[str]) -> bool:
    if not expected_keywords:
        return True
    lowered = answer_text.lower()
    return all(keyword.lower() in lowered for keyword in expected_keywords)


def write_report(results: list[dict], path: str) -> None:
    total = len(results)
    retrieval_hits = sum(1 for r in results if r["retrieval_hit"])
    answer_hits = sum(1 for r in results if r["answer_hit"])
    both_hits = sum(1 for r in results if r["retrieval_hit"] and r["answer_hit"])

    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Questions evaluated: {total}",
        f"- Retrieval hit-rate: {retrieval_hits}/{total}",
        f"- Answer coverage: {answer_hits}/{total}",
        f"- Overall pass (both): {both_hits}/{total}",
        "",
        "| ID | Type | Question | Retrieval | Answer | Overall |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        overall = "PASS" if r["retrieval_hit"] and r["answer_hit"] else "FAIL"
        lines.append(
            f"| {r['id']} | {r['type']} | {r['question']} | "
            f"{'✓' if r['retrieval_hit'] else '✗'} | {'✓' if r['answer_hit'] else '✗'} | {overall} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_eval_scoring.py -v`
Expected: all 7 PASS.

- [ ] **Step 6: Commit**

```bash
git add eval/qa_dataset.json eval/scoring.py tests/test_eval_scoring.py
git commit -m "feat: add curated RAG evaluation dataset and scoring functions"
```

---

### Task 11: eval/run_eval.py orchestration + generate the committed report

**Files:**
- Create: `eval/run_eval.py`
- Modify: `.gitignore`, `README.md`

**Interfaces:**
- Consumes: `index_document(file_path, document_id, vector_store, session_id)` (Task 2), `answer_question(vector_store, session_id, document_ids, question)` (Task 4), `score_retrieval_hit`, `score_answer_coverage`, `write_report` (Task 10).
- Produces: `run_eval(dataset, vector_store, document_ids_by_name) -> list[dict]` (orchestration, not separately unit tested — it makes real Groq API calls, same rationale as why `app.py` has no automated tests); a `main()` CLI entrypoint; the committed `eval/report.md` artifact this whole subsystem exists to produce.

- [ ] **Step 1: Implement the orchestration script**

Create `eval/run_eval.py`:

```python
import json
from pathlib import Path

from core.answer import answer_question
from core.indexing import index_document
from core.vectorstore import VectorStore
from eval.scoring import score_answer_coverage, score_retrieval_hit, write_report

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
DATASET_PATH = Path(__file__).parent / "qa_dataset.json"
REPORT_PATH = Path(__file__).parent / "report.md"
EVAL_STORE_DIR = Path(__file__).parent / "eval_chroma_db"
SESSION_ID = "eval"


def _index_referenced_documents(dataset: list[dict], vector_store: VectorStore) -> dict[str, str]:
    document_ids_by_name = {}
    for item in dataset:
        name = item["document"]
        if name in document_ids_by_name:
            continue
        document_id = f"eval-{len(document_ids_by_name)}"
        index_document(
            str(FIXTURES_DIR / name), document_id=document_id, vector_store=vector_store, session_id=SESSION_ID
        )
        document_ids_by_name[name] = document_id
    return document_ids_by_name


def run_eval(dataset: list[dict], vector_store: VectorStore, document_ids_by_name: dict[str, str]) -> list[dict]:
    results = []
    for item in dataset:
        document_id = document_ids_by_name[item["document"]]
        answer = answer_question(vector_store, SESSION_ID, [document_id], item["question"])
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "type": item["type"],
                "retrieval_hit": score_retrieval_hit(answer.sources, item["expected_locations"]),
                "answer_hit": score_answer_coverage(answer.text, item["expected_keywords"]),
            }
        )
    return results


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    vector_store = VectorStore(persist_directory=str(EVAL_STORE_DIR))
    document_ids_by_name = _index_referenced_documents(dataset, vector_store)
    results = run_eval(dataset, vector_store, document_ids_by_name)
    write_report(results, str(REPORT_PATH))

    passed = sum(1 for r in results if r["retrieval_hit"] and r["answer_hit"])
    print(f"{passed}/{len(results)} questions passed — report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Gitignore the eval script's own local vector store**

In `.gitignore`, add a line (near the existing `storage/chroma_db/` entry):

```
eval/eval_chroma_db/
```

- [ ] **Step 3: Run it for real**

Run, from the project root: `python -m eval.run_eval`

This makes real Groq API calls for all 20 questions (a couple of minutes, subject to free-tier rate limits — the existing retry/backoff in `core/llm.py` handles transient throttling). Expected: it prints something like `17/20 questions passed — report written to eval/report.md`, and `eval/report.md` now exists with the summary + full per-question table.

Read `eval/report.md`. For any FAILs, open the question and decide: is it a genuine pipeline weakness worth noting, or a too-strict/ambiguous `expected_keywords` choice in `eval/qa_dataset.json` (e.g. the model correctly answered but phrased it differently than the exact keyword)? This dataset was hand-scored specifically so you exercise this judgment — per `docs/06_PRD_PHASE2.md`, LLM-as-judge grading is explicitly out of scope. If a failure is clearly a dataset wording issue rather than a real pipeline problem, it's fine to adjust that entry's `expected_keywords` and re-run — but don't loosen a keyword just to force a pass on a question the pipeline genuinely got wrong.

- [ ] **Step 4: Update README with the eval command**

In `README.md`, add a short new subsection after "Running tests":

```
## Evaluating RAG quality
   python -m eval.run_eval

Runs a curated 20-question set (eval/qa_dataset.json) against the real
pipeline (real embeddings, real Groq calls) and writes eval/report.md with
retrieval hit-rate and answer-coverage scores per question. Not part of the
default test suite — it costs real API latency/rate-limit budget.
```

- [ ] **Step 5: Commit**

```bash
git add eval/run_eval.py eval/report.md .gitignore README.md
git commit -m "feat: add eval orchestration script and generate the first evaluation report"
```

---

## Summary of new/changed files

- `core/vectorstore.py`, `core/indexing.py`, `core/retriever.py`, `core/answer.py`, `core/chat_handler.py`, `app.py` — session isolation (Tasks 1-5)
- `core/ocr.py` (new), `core/ingest.py`, `packages.txt` (new), `run_steps.txt`, `README.md` — OCR (Tasks 6-7)
- `core/reranker.py` (new), `core/retriever.py` — reranking (Tasks 8-9)
- `eval/qa_dataset.json`, `eval/scoring.py`, `eval/run_eval.py`, `eval/report.md` (all new), `.gitignore`, `README.md` — evaluation dataset (Tasks 10-11)
