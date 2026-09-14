# PRD — AI PDF Document Assistant (RAG)

## 1. Summary
A web app where a user uploads a PDF (e.g. a 100-page contract) and asks natural-language questions about it. The app retrieves the relevant passages via a RAG pipeline and answers using an LLM, citing the page/section the answer came from.

Extension of an existing PDF Editor product. Built with no paid LLM subscription — uses Groq's free API tier.

## 2. Problem
Reading long PDFs (contracts, reports, manuals) to find specific clauses or answer specific questions is slow. Full-text search finds keywords, not answers. This tool lets a user ask a question in plain language and get a grounded, cited answer.

## 3. Goals (MVP)
- Upload a document — **PDF, DOCX, or TXT** (up to ~150 pages / ~20MB). Legacy `.doc` is explicitly rejected with a message to save as `.docx`.
- Extract and chunk the text
- Embed chunks and store them in a local vector index
- Retrieve top-k relevant chunks for a user question
- Generate an answer via an LLM (Groq), grounded only in retrieved chunks
- Show the answer with source page numbers
- Support follow-up questions in the same session (basic chat memory)

## 4. Non-goals (MVP)
- Multi-user accounts / auth
- Multi-document cross-referencing (v2)
- OCR for scanned/image-only PDFs (v2 — flag as unsupported for now)
- Legacy `.doc` (pre-2007 binary Word format) support
- Editing the PDF itself (that's the existing PDF Editor product's job)
- Fine-tuning any model

## 5. User stories
1. As a user, I upload a contract PDF and see confirmation it's been processed ("312 chunks indexed").
2. As a user, I type "What is the payment due date?" and get a direct answer with the page/clause it came from.
3. As a user, I ask "Summarize the termination conditions" and get a synthesized summary spanning multiple chunks, not just one match.
4. As a user, I ask "Find all references to penalties" and get every relevant instance listed, not just the single best match.
5. As a user, if I ask something not covered in the document, the app tells me it can't find that in the document instead of guessing.
6. As a user, I can ask a follow-up question ("what about late payment?") and the app understands it's related to the previous answer.

## 6. Key functional requirements
| # | Requirement |
|---|---|
| F1 | Accept PDF, DOCX, and TXT uploads; validate file type/size; reject `.doc` with a clear "save as .docx" message |
| F2 | Extract text with a location reference per chunk — page number for PDF, paragraph/heading index for DOCX, line-range for TXT |
| F3 | Chunk text (~500-800 tokens, with overlap) while keeping page-number metadata per chunk |
| F4 | Generate embeddings for each chunk, store in a vector index |
| F5 | On a question, embed the query, retrieve top-k similar chunks |
| F6 | For "find all references" style questions, retrieve a larger k and/or do a full-document scan, not just top-k |
| F7 | Build a prompt that includes retrieved chunks + page numbers + chat history, send to Groq LLM |
| F8 | Return answer with cited page numbers; if no relevant chunk found, say so explicitly |
| F9 | Maintain conversation history per uploaded document (session-scoped) |
| F10 | Handle PDFs with no extractable text (scanned) by warning the user |

## 7. Non-functional requirements
- Must run entirely on free-tier services (no paid OpenAI/Claude/Gemini key required)
- Answer latency: aim for under ~8s per question with Groq's fast inference
- Must not hallucinate beyond the document — answers should be grounded, with a visible "not found in document" fallback
- Local-first: PDFs and vector index stored on disk/local DB, not sent to third parties beyond the LLM call itself

## 8. Success metrics (informal, since this is a personal/portfolio project)
- Can correctly answer at least 8/10 manually-verified factual questions on a real 50-100 page contract
- "Find all references to X" returns all actual occurrences, not a subset
- No crash on a 100+ page PDF
