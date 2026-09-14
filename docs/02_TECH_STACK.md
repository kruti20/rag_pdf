# Tech stack — AI PDF Document Assistant

Constraint driving every choice below: **no paid LLM API, no paid vector DB, must run on a normal laptop.**

## Chosen stack

| Layer | Choice | Why |
|---|---|---|
| UI | **Streamlit** | Fastest way to get file upload + chat UI + working app in one Python file. No separate frontend build/deploy. |
| PDF text extraction | **PyMuPDF (`fitz`)** | Faster and more accurate layout/page handling than pypdf; gives per-page text easily. |
| DOCX text extraction | **`python-docx`** | Reads paragraphs and headings directly; free, no external binary needed. |
| TXT text extraction | **built-in `open()`** | Nothing to add — just read the file with encoding detection (`chardet` or `utf-8` with fallback). |
| Chunking | **LangChain `RecursiveCharacterTextSplitter`** (or hand-rolled equivalent) | Splits by paragraph/sentence boundaries first, falls back to characters. Keeps chunks semantically coherent. |
| Embeddings | **`sentence-transformers` — `all-MiniLM-L6-v2`** (runs locally, CPU is fine) | Free, no API key, no rate limit, good enough quality for this use case, ~80MB model. |
| Vector store | **ChromaDB** (local, embedded, persists to disk) | Zero-config, no server to run, free, integrates directly with Python. |
| LLM (generation) | **Groq API — free tier** (model: `llama-3.3-70b-versatile`, fallback `llama-3.1-8b-instant` for speed) | No credit card required, extremely fast inference, generous enough free rate limits for a personal app. Model availability/limits shift over time — check `console.groq.com` before hardcoding a model id. |
| Orchestration | **LangChain** (optional) or plain Python functions | LangChain saves boilerplate for chunking/retrieval chains, but plain Python keeps you in control and is easier to debug when free-tier things go wrong (rate limits, empty responses). Recommendation: start with plain Python, adopt LangChain only if it clearly saves time. |
| Session/chat memory | In-memory Python list per Streamlit session (`st.session_state`) | No DB needed for MVP; resets per session, which is fine for a single-user tool. |

## Why NOT these alternatives (for now)
- **OpenAI / Claude / Gemini paid APIs** — ruled out by the "no pro model" constraint.
- **Gemini free tier** — viable alternative to Groq, but Groq's inference speed is a better fit for a snappy chat UX; keep Gemini as backup if Groq rate limits are too tight.
- **Ollama (fully local LLM)** — great if you have a decent GPU/RAM (16GB+); slower and heavier to set up than an API call. Good v2 option for full offline/privacy mode.
- **Pinecone / Weaviate cloud** — unnecessary network dependency and paid tiers for a single-user app; ChromaDB local is simpler and free.
- **FastAPI + React** — more "real" product architecture, but slower to build. Revisit once the RAG pipeline itself is proven (see Architecture doc's "decouple core from UI" note).

## requirements.txt (starting point)
```
streamlit
pymupdf
python-docx
chardet
sentence-transformers
chromadb
groq
langchain-text-splitters
python-dotenv
```

## Environment variables
```
GROQ_API_KEY=your_free_groq_key   # from console.groq.com
```

## Known free-tier limits to design around
- Groq free tier is rate-limited per minute and per day (exact numbers change — check your account dashboard). Design implication: batch retrieval so you make **one** LLM call per question, not one per chunk.
- `all-MiniLM-L6-v2` embeddings run on CPU with no external call, so embedding is not rate-limited — only the final generation step depends on Groq.
