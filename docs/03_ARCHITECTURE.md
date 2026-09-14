# Architecture — AI PDF Document Assistant

## 1. Design principle
Keep the **RAG core** (extraction → chunking → embedding → retrieval → generation) as plain, UI-agnostic Python modules. Streamlit only calls into this core. This means:
- The core is testable without a browser.
- Swapping Streamlit for FastAPI + React later is a UI-layer swap, not a rewrite.

## 2. Pipeline (matches the flow in the brief)

```
Upload document (PDF / DOCX / TXT)
   → [ingest.py]      dispatch by file extension, extract text + location metadata
                         PDF  → PyMuPDF, location = page_number
                         DOCX → python-docx, location = paragraph_index (+ nearest heading)
                         TXT  → plain read, location = line_range
   → [chunker.py]     split into overlapping chunks, keep location metadata
   → [embedder.py]    embed each chunk (sentence-transformers, local)
   → [vectorstore.py] upsert into ChromaDB collection (one collection per document)

Ask question
   → [embedder.py]    embed the question
   → [vectorstore.py] similarity search → top-k chunks (+ page numbers)
   → [prompt.py]      build prompt: system instructions + retrieved chunks + chat history + question
   → [llm.py]         call Groq chat completion
   → [answer.py]      parse response, attach source page numbers, return to UI
```

## 3. Special handling per question type
- **Direct fact ("payment due date?")** → standard top-k (k≈4-6) similarity search, single LLM call.
- **Summarization ("summarize termination conditions")** → retrieve a larger k (e.g. 10-15) filtered to chunks matching "termination"-related terms via keyword pre-filter + semantic search combined, then ask the LLM to synthesize across all of them.
- **Exhaustive search ("find all references to penalties")** → don't rely on top-k alone (it can miss occurrences). Do a keyword scan across all chunks for "penalt*" first, union with semantic top-k, then pass all matches to the LLM to compile a list with page numbers. This hybrid (keyword + semantic) is important — pure vector search can silently drop instances.

## 4. Folder structure
```
pdf-rag-assistant/
├── app.py                  # Streamlit entrypoint (UI only)
├── core/
│   ├── ingest.py            # PDF/DOCX/TXT → text with location metadata (one loader function per type)
│   ├── chunker.py           # text → chunks with metadata
│   ├── embedder.py          # text → vector
│   ├── vectorstore.py       # Chroma wrapper: add/query per document
│   ├── retriever.py         # hybrid keyword+semantic retrieval, question-type routing
│   ├── prompt.py            # prompt templates
│   ├── llm.py               # Groq client wrapper (with retry/backoff for rate limits)
│   └── models.py            # dataclasses: Chunk, Source, Answer
├── storage/
│   └── chroma_db/           # persisted vector DB (gitignored)
├── tests/
│   ├── test_chunker.py
│   ├── test_retriever.py
│   └── fixtures/sample.pdf
├── requirements.txt
├── .env.example
└── README.md
```

## 5. Data model
```python
Chunk:
  id: str
  document_id: str
  source_type: str        # "pdf" | "docx" | "txt"
  location_label: str     # e.g. "p.44", "para 12 (Termination)", "lines 210-240"
  text: str
  embedding: list[float]

Source:
  location_label: str
  snippet: str

Answer:
  text: str
  sources: list[Source]
  found_in_document: bool
```

## 6. Key design decisions & why
- **One Chroma collection per uploaded document** — keeps documents isolated, simple to clear/reset, avoids cross-document leakage in answers.
- **Location label as first-class metadata, format-aware** — required for citations; attach at chunk-creation time, never re-derive later. PDFs cite by page, DOCX by paragraph/nearest heading, TXT by line range — `ingest.py` normalizes all three into one `location_label` string so the rest of the pipeline doesn't need to know which file type it came from.
- **`.doc` rejected at upload, not silently mishandled** — validate extension immediately and show a clear message rather than attempting a fragile conversion.
- **Hybrid retrieval, not pure vector search** — pure semantic top-k is the most common RAG failure mode for "find all X" queries; a keyword pass catches exact-term mentions vector search can rank low.
- **Explicit "not found" path** — prompt instructs the LLM to say "I couldn't find that in the document" rather than answer from general knowledge, to avoid hallucination. This should be a hard instruction in the system prompt, and ideally also checked by confirming retrieved chunks pass a minimum similarity threshold before calling the LLM at all.
- **Session-scoped memory only for MVP** — no persistence of chat history needed yet; keep it simple.

## 7. Future path (v2, not in this MVP scope)
- Swap Streamlit UI for FastAPI backend + React frontend (reuse `core/` unchanged)
- Multi-document / cross-document Q&A
- OCR fallback for scanned PDFs (e.g. `pytesseract`)
- Persistent chat history per document (SQLite)
- Swap ChromaDB for a hosted vector DB if scaling beyond single-user
