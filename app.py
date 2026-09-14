import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.chat_handler import build_answer_message
from core.indexing import index_document
from core.ingest import DocumentValidationError
from core.vectorstore import VectorStore

st.set_page_config(page_title="AI PDF Document Assistant", page_icon="📄", layout="wide")

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


if "document_id" not in st.session_state:
    st.session_state.document_id = None
    st.session_state.document_name = None
    reset_chat()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.title("📄 AI PDF Document Assistant")

left, right = st.columns([1, 2])

with left:
    st.subheader("Document")
    uploaded_file = st.file_uploader("Drag a PDF, Word (.docx), or text file here", type=None)

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        document_id = hashlib.sha256(file_bytes).hexdigest()[:16]

        if document_id != st.session_state.document_id:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                with st.spinner("Extracting text… Chunking… Embedding…"):
                    result = index_document(
                        tmp_path, document_id=document_id, vector_store=get_vector_store()
                    )
            except DocumentValidationError as e:
                st.error(str(e))
            else:
                st.session_state.document_id = document_id
                st.session_state.document_name = uploaded_file.name
                st.session_state.source_type = result.source_type
                st.session_state.section_count = len(result.segments)
                st.session_state.chunk_count = result.chunk_count
                st.session_state.has_extractable_text = result.has_extractable_text
                reset_chat()

    if st.session_state.document_id:
        st.markdown(f"📄 **{st.session_state.document_name}**")
        size_label = SIZE_LABELS.get(st.session_state.source_type, "sections")
        st.caption(f"{st.session_state.section_count} {size_label} · {st.session_state.source_type.upper()}")

        if not st.session_state.has_extractable_text:
            st.warning(
                "This document looks scanned or has no extractable text — "
                "answers may be limited or unavailable."
            )
        else:
            st.success(f"indexed — {st.session_state.chunk_count} chunks")

        st.markdown("**Suggested questions:**")
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, key=f"suggested-{question}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

with right:
    st.subheader("Chat")

    if not st.session_state.document_id:
        st.info("Upload a PDF, Word (.docx), or text file to get started.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and not message.get("found_in_document", True):
                st.markdown(f"*{message['text']}*")
            else:
                st.markdown(message["text"])
                if message.get("sources"):
                    st.caption("Sources: " + ", ".join(message["sources"]))

    document_ready = st.session_state.document_id is not None
    question = st.chat_input(
        "Ask a question about this document" if document_ready else "Upload a document to get started",
        disabled=not document_ready,
    )
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question and document_ready:
        st.session_state.chat_history.append({"role": "user", "text": question})
        history_for_prompt = [
            (m["role"], m["text"]) for m in st.session_state.chat_history[:-1]
        ]

        with st.spinner("Searching document…"):
            message = build_answer_message(
                get_vector_store(),
                st.session_state.document_id,
                question,
                chat_history=history_for_prompt,
            )
            st.session_state.chat_history.append(message)
        st.rerun()
