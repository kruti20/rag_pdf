# Implementation plan — AI PDF Document Assistant

Feed this whole document set (01-05) into Claude Code. Suggested prompt to start:
> "Read all 5 planning docs in this folder. Build the project following the folder structure in 03_ARCHITECTURE.md, starting with Phase 1."

Work through phases in order; each should end in something runnable.

## Phase 0 — Project setup
- [ ] Create folder structure from `03_ARCHITECTURE.md`
- [ ] `requirements.txt` from `02_TECH_STACK.md`, set up virtualenv
- [ ] `.env.example` with `GROQ_API_KEY`
- [ ] Get a free Groq API key (console.groq.com), confirm a simple test call works
- [ ] `README.md` with setup + run instructions

## Phase 1 — Document ingestion + chunking (no LLM yet)
- [ ] `core/ingest.py`: one loader per file type, each returning normalized text + `location_label`
      - PDF: PyMuPDF, per-page text, `location_label = "p.{n}"`
      - DOCX: `python-docx`, per-paragraph text, track nearest heading, `location_label = "para {n} ({heading})"`
      - TXT: plain read, chunk by line ranges, `location_label = "lines {a}-{b}"`
- [ ] File-type dispatch + validation: accept `.pdf/.docx/.txt`, reject `.doc` and anything else with a clear message
- [ ] `core/chunker.py`: split extracted text into overlapping chunks (~500-800 tokens, ~50-100 token overlap), carry `location_label` and `document_id` through to each chunk
- [ ] `core/models.py`: `Chunk` dataclass (see `03_ARCHITECTURE.md` data model)
- [ ] Unit test: run against a sample PDF, DOCX, and TXT file each; verify chunk count and location labels look right
- [ ] Handle edge case: PDF with no extractable text (scanned) → return empty result with a flag, don't crash

## Phase 2 — Embeddings + vector store
- [ ] `core/embedder.py`: load `all-MiniLM-L6-v2`, function to embed a list of texts
- [ ] `core/vectorstore.py`: Chroma wrapper — create/get collection per `document_id`, `add_chunks()`, `query(embedding, k)`
- [ ] Wire Phase 1 + 2 together: upload → extract → chunk → embed → store
- [ ] Test: query a known phrase from the sample PDF, confirm the right chunk/page comes back

## Phase 3 — Retrieval logic
- [ ] `core/retriever.py`: semantic top-k retrieval function
- [ ] Add keyword-match retrieval (simple substring/regex scan across all chunks) for the "find all references" case
- [ ] Add simple question-type detection (factual vs summarization vs "find all") — can start as keyword heuristics ("summarize", "list all", "find all", "every reference") before anything fancier
- [ ] Combine hybrid results, dedupe by chunk id

## Phase 4 — LLM integration
- [ ] `core/llm.py`: Groq client wrapper, function `generate(prompt) -> str`, with retry/backoff on rate-limit errors
- [ ] `core/prompt.py`: system prompt enforcing "only answer from provided context; say 'not found in document' if not covered", template that inserts retrieved chunks (with page numbers) + chat history + question
- [ ] `core/answer.py`: call retriever → build prompt → call LLM → parse into `Answer` (text + sources + found_in_document)
- [ ] Manual test: ask the 3 example questions from the brief against a real contract PDF, verify answers and citations

## Phase 5 — Streamlit UI
- [ ] `app.py`: layout per `04_UI_UX_SPEC.md` — two columns, file uploader, chat interface
- [ ] Wire upload → ingestion pipeline with progress indicator
- [ ] Wire chat input → `core/answer.py`, render answer + sources
- [ ] Suggested question buttons
- [ ] Empty/loading/error/not-found states from the UX spec
- [ ] Session state for chat history per uploaded document

## Phase 6 — Polish & edge cases
- [ ] File size / type validation with clear error messages
- [ ] Scanned-PDF detection (near-zero extracted text) → warn user, don't silently fail
- [ ] Rate-limit handling surfaced nicely in the UI
- [ ] Basic tests for chunker and retriever (`tests/`)
- [ ] README: how to run, how to get a Groq key, known limitations

## Explicitly out of scope for this build (see PRD "non-goals")
- Auth / multi-user
- OCR
- Multi-document Q&A
- Persistent chat history across sessions
