# AI PDF Document Assistant

A RAG app for asking natural-language questions about a PDF, DOCX, or TXT document, with
answers grounded in the document and cited by page/paragraph/line. Runs entirely on free-tier
services — no paid LLM API key required.

See `docs/` for the full product/tech/architecture spec this project was built from.

## Stack
Streamlit · PyMuPDF · python-docx · sentence-transformers (`all-MiniLM-L6-v2`) · ChromaDB · Groq (free tier, `openai/gpt-oss-120b`)

Note: Groq's free-tier model lineup changes over time. If `core/llm.py`'s model id ever
404s, check `console.groq.com` (or `client.models.list()`) for current model names.

## Setup

1. **Python**: requires Python 3.11 (newer versions may lack prebuilt wheels for some ML deps).

2. **Create and activate a virtualenv**
   ```bash
   python -m venv venv
   # Windows (git-bash):
   source venv/Scripts/activate
   # Windows (PowerShell):
   venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   On Windows, if `import torch` fails with `OSError: ... shm.dll or one of its dependencies`,
   the pulled-in torch build is broken for your setup — reinstall the pinned CPU build:
   ```bash
   pip uninstall -y torch
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Get a free Groq API key**
   - Sign up at https://console.groq.com
   - Create an API key
   - Copy `.env.example` to `.env` and set `GROQ_API_KEY=your_key_here`

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Running tests
```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Project layout
```
app.py                  # Streamlit UI (thin layer over core/)
core/                   # UI-agnostic RAG pipeline — ingest, chunk, embed, retrieve, prompt, LLM call
storage/chroma_db/      # persisted local vector DB (gitignored)
tests/                  # unit tests + sample fixtures
```

## Known limitations (MVP)
- No OCR — scanned/image-only PDFs will show a warning, not extracted text
- Legacy `.doc` files are rejected — save as `.docx` first
- Files over 20MB are rejected with a clear message (per the PRD's ~150 page / ~20MB target)
- Corrupted, malformed, or password-protected PDFs/DOCX are rejected with a clear message
  instead of crashing (scanned-but-readable PDFs still index — see the OCR note above)
- Single-document sessions only (no cross-document Q&A)
- Chat history is session-scoped only (not persisted across restarts)
- Groq free-tier rate limits apply — `core/llm.py` retries with backoff internally; if all
  retries are exhausted, the chat shows a friendly "hit the rate limit" message rather than
  a stack trace
