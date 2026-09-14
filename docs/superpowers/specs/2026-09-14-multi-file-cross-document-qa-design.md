# Design: Multi-File Upload with Cross-Document Q&A

Date: 2026-09-14

## 1. Summary

Extend the app from a single-document session to a library of up to 10
documents. Factual and "find all references" questions are always answered
across the entire library, with citations naming which document each excerpt
came from. Summarization questions are the one exception: when more than one
document is loaded, the app asks which single document to summarize before
answering. This supersedes the PRD's "multi-document cross-referencing"
non-goal.

## 2. Goals

- Upload up to 10 PDF/DOCX/TXT files into a session library.
- Reject an 11th upload with a clear error until a slot is freed.
- Let the user remove any document from the library (frees its slot and its
  vector store collection).
- Factual and exhaustive ("find all references") questions automatically
  search the whole library — no manual scope selection.
- Citations name their source document (e.g. "ContractA.pdf — Page 5").
- When a summarization-type question is asked and more than one document is
  loaded, ask the user which single document to summarize before answering;
  skip the prompt if only one document is loaded.
- Keep factual-question cost (chunks sent to the LLM) constant regardless of
  library size, by applying existing top-k caps globally across the merged
  result set rather than per document.

## 3. Non-goals

- No manual subset/checkbox selection for factual or exhaustive questions —
  those always span the full library.
- No cap on "find all references to X" style queries across documents — see
  §7 trade-off.
- No persistence of the library across app restarts (still session-scoped,
  matching existing chat-history behavior).
- No multi-document combined summary (e.g. "summarize all of these into
  one") — summarization is always scoped to exactly one document, chosen by
  the user when the library has more than one.

## 4. State model (`app.py`)

```
st.session_state.documents: dict[document_id -> {
    name: str,
    source_type: str,
    section_count: int,
    chunk_count: int,
    has_extractable_text: bool,
}]
st.session_state.pending_summarize_question: str | None  # new
```

No `selected` field, no active-document concept, no scope-change tracking.
Chat history persists for the session and is not reset when documents are
added or removed. The library dict preserves insertion order for display.

## 5. Upload & library UI (`app.py`)

- `st.file_uploader(..., accept_multiple_files=True)` is used purely as an
  "add to library" input. The persisted library lives in
  `st.session_state.documents`, independent of what the widget currently
  displays.
- For each file returned by the uploader on a rerun:
  - Hash it (existing sha256 approach) → `document_id`. Skip if already in
    the library.
  - If `len(documents) >= 10`: show
    `st.error(f"Maximum 10 documents — remove one before adding '{file.name}'")`
    and skip this file; other files in the same batch still process.
  - Otherwise run the existing `index_document(...)` unchanged. On
    `DocumentValidationError`, show an error scoped to that filename and
    skip it. On success, add to `documents`.
- Library rows: header `Documents (n/10)`; each row shows the name,
  per-document status (chunk count, source type, scanned-doc warning if
  `not has_extractable_text`), and a 🗑 remove button.
- Removing a document: call `VectorStore.delete_document(document_id)` and
  delete it from `documents`.
- Chat input is disabled with "Upload a document to get started" when the
  library is empty (same message/condition as today, just keyed off
  `len(documents) > 0` instead of a single `document_id`).

## 6. Question routing & the summarization clarification flow

- On every submitted question, first call the existing
  `detect_question_type(question)` (from `core/retriever.py`) in `app.py`.
- **Factual / exhaustive:** answer immediately with
  `document_ids = list(st.session_state.documents.keys())` — the full
  library, every time.
- **Summarization:**
  - If `len(documents) <= 1`: answer immediately, scoped to that one
    document (or normally if the library is empty — chat input is disabled
    in that case anyway).
  - If `len(documents) > 1`: don't answer yet. Append the user's question to
    chat history, store the raw question text in
    `st.session_state.pending_summarize_question`, and set a flag so the
    next chat render shows an inline picker instead of an assistant answer.
- **Inline picker (chat thread):** rendered as the pending assistant turn —
  a `st.radio` (or `st.selectbox`) listing current library document names,
  plus a "Summarize" button. On click, call `build_answer_message(...)` with
  `document_ids = [chosen_id]`, append the assistant's answer to chat
  history, clear `pending_summarize_question`, and rerun.
- If the user uploads or removes a document while a summarize picker is
  pending, the picker's option list simply reflects the current library on
  next rerun (no special-casing needed beyond re-reading `documents`).

## 7. Core changes

Signatures change from a single `document_id: str` to `document_ids:
list[str]`, plus a `document_names: dict[id, name]` map for citation/prompt
labeling. Chunk IDs are already globally unique (`f"{document_id}-{index}"`
per `core/chunker.py`), so merging results across documents is safe to dedupe
by ID. These functions don't know or care whether `document_ids` represents
the whole library or a single document chosen via the summarize picker.

- **`core/vectorstore.py`**: add `delete_document(document_id)` →
  `self.client.delete_collection(name=f"doc_{document_id}")`. `query()` and
  `get_all_chunks()` keep their existing single-document signature; they are
  now called in a loop instead of once.
- **`core/retriever.py`**: `retrieve(vector_store, document_ids, question)`
  loops over each id, tags every result dict with `document_id`, merges
  across documents, then applies a **global** cap:
  - factual: merge all documents' `query(k=FACTUAL_K)` results, sort by
    distance, keep global top `FACTUAL_K`.
  - exhaustive/summarization: merge all documents' `query(k=BROAD_K)`
    results, sort by distance, keep global top `BROAD_K`; keyword hits from
    `get_all_chunks()` across all documents are kept in full (see trade-off
    below), then combined and deduped by ID as today. (For summarization,
    `document_ids` will be a single-element list by the time this runs, per
    §6, so this path only ever merges within one document in practice.)
- **`core/models.py`**: `Source` gains `document_id: str`.
- **`core/prompt.py`**: `build_prompt(question, chunks, document_names,
  chat_history=None)`; `_format_chunk` becomes
  `f"[{document_names[chunk['document_id']]} — {chunk['location_label']}] {chunk['text']}"`
  so the LLM can distinguish which document each excerpt is from.
- **`core/answer.py`**: `answer_question(vector_store, document_ids,
  question, document_names, llm=None, chat_history=None)`; `Source`
  construction includes `document_id=c["document_id"]`.
- **`core/chat_handler.py`**: `build_answer_message(vector_store,
  document_ids, question, document_names, chat_history=None, answer_fn=...)`;
  citation strings become `f"{document_names[s.document_id]} — {s.location_label}"`.

## 8. Efficiency trade-off (explicit, not hidden)

- Factual questions cost the same regardless of library size: one query
  embedding, up to 10 fast local Chroma reads (milliseconds each, no
  network), globally capped at `FACTUAL_K` chunks sent to the LLM.
- "Find all references to X" across the full library is **not** capped
  beyond today's per-document behavior — the keyword-search component
  returns every real match across every document, by design, since the
  app's core requirement (PRD F6) is "don't miss instances." A broad
  exhaustive query over many large documents will therefore cost more tokens
  and take longer than the same query over one document. This is a
  conscious choice, not an oversight — an artificial cap would silently drop
  legitimate matches.
- Summarization is naturally bounded — it's always scoped to exactly one
  document (§6), so its cost never grows with library size.

## 9. Testing

- Update `tests/test_retriever.py`, `tests/test_answer.py`,
  `tests/test_prompt.py`, `tests/test_chat_handler.py` for the new
  list-based signatures and cross-document merge/cap behavior.
- Add a `VectorStore.delete_document` test to `tests/test_vectorstore.py`
  (add chunks → delete → `get_all_chunks` returns empty).
- `app.py` has no existing unit tests (Streamlit UI glue) and this design
  doesn't change that convention — verify the upload/library/summarize-picker
  flow by running the app manually.

## 10. Docs

- Update `run_steps.txt`'s "Single-document sessions only" limitation line
  to describe the new library + cross-document behavior, including the
  summarize-picker exception.
