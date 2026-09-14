# UI/UX spec — AI PDF Document Assistant

Framework: **Streamlit**, single page, two-column layout.

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│  📄  AI PDF Document Assistant                                │
├───────────────────────────┬───────────────────────────────────┤
│  DOCUMENT PANEL (left)    │  CHAT PANEL (right)                │
│                           │                                     │
│  [ Drag PDF here or       │  ┌─────────────────────────────┐   │
│    click to upload ]      │  │ (user) Summarize termination │   │
│                           │  │        conditions.            │   │
│  📄 contract_v2.pdf        │  └─────────────────────────────┘   │
│     104 pages              │  ┌─────────────────────────────┐   │
│  ✅ indexed — 312 chunks   │  │ (assistant) Either party may  │   │
│                           │  │ terminate with 30 days written│   │
│  Suggested questions:      │  │ notice...                     │   │
│  • What is the payment     │  │ Sources: p.44 §9.2, p.45 §9.4 │   │
│    due date?                │  └─────────────────────────────┘   │
│  • Summarize termination    │                                     │
│    conditions               │  [ Ask a question about this PDF ] │
│  • Find all references to  │  [Send]                             │
│    penalties                │                                     │
└───────────────────────────┴───────────────────────────────────┘
```

## Components
1. **Header** — app name + icon.
2. **Document panel**
   - Upload widget (`st.file_uploader`, accepts `.pdf`, `.docx`, `.txt`; size limit shown)
   - If a `.doc` file is dropped: reject immediately with "Old Word format (.doc) isn't supported — please save as .docx and re-upload."
   - Once uploaded: filename, file type badge, size indicator (page count for PDF, paragraph count for DOCX, line count for TXT), indexing status (spinner → checkmark with chunk count)
   - Suggested questions (clickable buttons that pre-fill the chat input) — generate 3-5 generic prompts once indexed (e.g. "Summarize this document", "What are the key dates?", "List any obligations or penalties")
   - Error state: if no extractable text found, show a warning: "This PDF looks scanned — text extraction may be incomplete."
3. **Chat panel**
   - Scrollable message history (`st.chat_message` for user/assistant bubbles)
   - Each assistant answer shows a **Sources** line: format-appropriate location labels (page numbers for PDF, paragraph/heading for DOCX, line range for TXT), clickable to jump to / highlight that location (v2: open a preview)
   - Text input pinned at the bottom (`st.chat_input`)
   - Loading state while waiting on the LLM (`st.spinner("Searching document...")`)
   - Explicit "not found" style answer rendered distinctly (e.g. muted/italic) so it's visually clear from a grounded answer

## States to design for
| State | Behavior |
|---|---|
| No document uploaded | Chat input disabled, show empty-state message: "Upload a PDF, Word (.docx), or text file to get started." |
| Unsupported file (.doc, .rtf, image, etc.) | Reject at upload with a specific message naming what's wrong and what to do |
| Uploading/indexing | Spinner with progress if possible ("Extracting text… Chunking… Embedding…") |
| Ready | Chat enabled, suggested questions visible |
| Answer found | Bubble + sources |
| Answer not found in doc | Bubble clearly marked as "not found", no fake sources |
| Error (corrupt PDF, no text, API failure) | Clear inline error message, never a raw stack trace |
| Rate limit hit (Groq free tier) | Friendly message: "Hit the free-tier rate limit, retrying in a few seconds…" with automatic retry/backoff |

## Interaction notes
- Clicking a suggested question fills and submits the chat input — should feel instant.
- Sources should always be visible when present; never hide citations behind a click for MVP.
- Keep the whole thing on one page — no multi-page nav needed for MVP.
