# PRD — Phase 2 (Production Hardening + RAG Quality)

## 1. Summary
Phase 1 shipped a feature-complete single/multi-document RAG assistant (see `01_PRD.md`, now built and deployed to Streamlit Community Cloud). Phase 2 addresses the four highest-priority gaps identified in a portfolio review of that deployment: a real multi-visitor data-isolation bug, a documented non-goal (OCR) now worth closing, and two RAG-quality improvements that demonstrate deeper AI-engineering skill to a technical reviewer.

This phase is scoped to exactly four items. Orange-priority items (better source viewer, streaming responses, question rewriting, architecture diagram) and green-priority items (README improvements, demo video) are explicitly deferred to a later phase.

## 2. Problem
The deployed app has a real architectural bug: `VectorStore` is a single `@st.cache_resource` instance shared by every visitor to the deployed process, while the document *library* (`st.session_state.documents`) is per-browser-session. Two visitors can collide — one visitor's 🗑 Remove can delete a Chroma collection another visitor's session still references. This is exactly the kind of issue a technical reviewer would flag as "not actually production-safe," despite the project having no auth.

Separately, three gaps limit what the project demonstrates: scanned PDFs are explicitly unsupported (a documented Phase-1 non-goal), retrieval ranks purely by embedding distance with no second-pass relevance check, and there is no artifact demonstrating the pipeline's answer quality was ever systematically measured.

## 3. Goals (Phase 2)
- **Session isolation**: each browser session's documents live in a fully separate slice of the shared vector store — no cross-visitor interference, no auth required.
- **OCR**: scanned/image-only PDF pages are transcribed automatically via Tesseract when normal text extraction finds nothing, using only free/open-source tooling.
- **RAG evaluation dataset**: a curated, hand-scored set of questions against the real fixture documents, run through the real pipeline, producing a human-readable quality report.
- **Retrieval reranking**: factual-question retrieval gets a cross-encoder second pass over a widened candidate pool before the final top-k is chosen, improving ranking precision over raw embedding distance alone.

## 4. Non-goals (Phase 2)
- Authentication / persistent user identity (session isolation is per-browser-session, not per-login — unchanged from Phase 1's non-goal).
- Cleanup/garbage-collection of orphaned per-session Chroma collections. They accumulate until the Streamlit Cloud process restarts (which already happens periodically since its filesystem is ephemeral); this is an accepted limitation, not solved this phase.
- Reranking on the exhaustive/summarization hybrid retrieval path (its goal is breadth, not precision-ranked top-k — reranking doesn't fit).
- LLM-as-judge or any automated grading for the evaluation dataset — pairs are hand-scored against curated expected keywords/locations.
- Wiring the evaluation suite into the default `pytest tests/` run or CI (it makes real Groq API calls and costs latency/rate-limit budget; it stays a manually-run script).
- Solving the memory-footprint increase from adding OCR + a reranker model preemptively. We measure after implementing, not before.
- Orange/green priority items (source viewer, streaming, question rewriting, architecture diagram, README improvements, demo video) — a later phase.

## 5. User stories
1. As a visitor to the deployed app, my uploaded documents and questions never interfere with another visitor's, even if we're using the app at the same time.
2. As a user, if I upload a scanned contract, the app extracts and indexes its text instead of only warning me it can't.
3. As the project owner, I can run one command and get a report showing the pipeline's measured answer accuracy against a real question set — something I can point to in an interview.
4. As a user asking a specific factual question, the citation I get back is the most relevant match available, not just whichever chunk happened to have the smallest raw embedding distance.

## 6. Key functional requirements
| # | Requirement |
|---|---|
| P2-F1 | Generate a random `session_id` per browser session on first load; thread it through every `VectorStore` collection name (`doc_{session_id}_{document_id}`) |
| P2-F2 | `VectorStore` methods (`add_chunks`, `query`, `get_all_chunks`, `delete_document`) accept/require a session-scoping value so two sessions with the same `document_id` never share a collection |
| P2-F3 | `core/ingest.py`'s PDF loader: when a page's normal text extraction is empty, render the page to an image and run Tesseract OCR on it before giving up on that page |
| P2-F4 | OCR is per-page and automatic — a text-based PDF never invokes OCR; only pages that yield no extractable text do |
| P2-F5 | New `eval/qa_dataset.json`: curated question set against real `tests/fixtures/*.pdf` documents, each entry with `question`, `expected_keywords`, `expected_location` |
| P2-F6 | New `eval/run_eval.py`: runs each question through the real pipeline, scores retrieval hit-rate (expected location present in retrieved chunks) and answer coverage (expected keywords present in generated answer), writes `eval/report.md` |
| P2-F7 | New `core/reranker.py`: wraps a `sentence-transformers` `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with a `rerank(question, candidates) -> list[dict]` function |
| P2-F8 | `core/retriever.py`'s factual path: widen the initial per-document semantic fetch (e.g. top-12), rerank the merged candidate pool, then take the final top-5 by rerank score |

## 7. Non-functional requirements
- Must continue to run entirely on free-tier services (Tesseract is open-source/local; the reranker is a local model — no new paid API).
- Session isolation must not require any new user-visible step (no login, no manual ID entry).
- OCR failure (e.g. Tesseract not installed in some environment) must degrade to the existing "looks scanned" warning, not crash ingestion.
- Memory footprint increase from OCR + reranker is a known risk on the free-tier deployment; measured and documented after implementation, not blocked on in advance.

## 8. Success metrics (informal — portfolio project)
- Two simultaneous browser sessions uploading different documents never see each other's documents disappear or leak into results.
- A real scanned PDF (image-only pages) produces extracted, indexed, queryable text instead of the "looks scanned" warning.
- `eval/report.md` shows a measured pass rate (e.g. "24/28 questions correct") against the curated question set, committed to the repo as a visible artifact.
- On a sample of factual questions with a known best-matching chunk that is *not* the closest by raw embedding distance, reranking surfaces it in the final top-5 where the unranked baseline did not.
