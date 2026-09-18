import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.chat_handler import build_answer_message
from core.indexing import index_document
from core.ingest import DocumentValidationError
from core.retriever import detect_question_type
from core.vectorstore import VectorStore

st.set_page_config(page_title="AI PDF Document Assistant", page_icon="📄", layout="wide")

MAX_DOCUMENTS = 10

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


if "documents" not in st.session_state:
    st.session_state.documents = {}
    reset_chat()
if "removed_document_ids" not in st.session_state:
    st.session_state.removed_document_ids = set()
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "pending_summarize_question" not in st.session_state:
    st.session_state.pending_summarize_question = None

st.title("📄 AI PDF Document Assistant")

left, right = st.columns([1, 2])

with left:
    uploaded_files = st.file_uploader(
        "Drag PDF, Word (.docx), or text files here", type=None, accept_multiple_files=True
    )

    selected_files = [
        (hashlib.sha256(f.getvalue()).hexdigest()[:16], f) for f in uploaded_files or []
    ]
    st.session_state.removed_document_ids &= {doc_id for doc_id, _ in selected_files}

    new_files = [
        (doc_id, f)
        for doc_id, f in selected_files
        if doc_id not in st.session_state.documents
        and doc_id not in st.session_state.removed_document_ids
    ]
    remaining_capacity = max(0, MAX_DOCUMENTS - len(st.session_state.documents))
    files_to_index, overflow_files = new_files[:remaining_capacity], new_files[remaining_capacity:]

    if overflow_files:
        st.error(
            f"Maximum {MAX_DOCUMENTS} documents allowed — remove one before adding: "
            + ", ".join(f"'{f.name}'" for _, f in overflow_files)
        )

    for document_id, uploaded_file in files_to_index:
        file_bytes = uploaded_file.getvalue()

        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                result = index_document(
                    tmp_path, document_id=document_id, vector_store=get_vector_store()
                )
        except DocumentValidationError as e:
            st.error(f"{uploaded_file.name}: {e}")
        else:
            st.session_state.documents[document_id] = {
                "name": uploaded_file.name,
                "source_type": result.source_type,
                "section_count": len(result.segments),
                "chunk_count": result.chunk_count,
                "has_extractable_text": result.has_extractable_text,
            }

    st.subheader(f"Documents ({len(st.session_state.documents)}/{MAX_DOCUMENTS})")

    for document_id, meta in list(st.session_state.documents.items()):
        with st.container(border=True):
            st.markdown(f"**{meta['name']}**")
            size_label = SIZE_LABELS.get(meta["source_type"], "sections")
            st.caption(f"{meta['section_count']} {size_label} · {meta['source_type'].upper()}")

            if not meta["has_extractable_text"]:
                st.warning(
                    "This document looks scanned or has no extractable text — "
                    "answers may be limited or unavailable."
                )
            else:
                st.success(f"indexed — {meta['chunk_count']} chunks")

            if st.button("🗑 Remove", key=f"remove-{document_id}"):
                get_vector_store().delete_document(document_id)
                del st.session_state.documents[document_id]
                st.session_state.removed_document_ids.add(document_id)
                st.rerun()

    if st.session_state.documents and not st.session_state.pending_summarize_question:
        st.markdown("**Suggested questions:**")
        for question in SUGGESTED_QUESTIONS:
            if st.button(question, key=f"suggested-{question}", use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

with right:
    st.subheader("Chat")

    if not st.session_state.documents:
        st.info("Upload a PDF, Word (.docx), or text file to get started.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and not message.get("found_in_document", True):
                st.markdown(f"*{message['text']}*")
            else:
                st.markdown(message["text"])
                if message.get("sources"):
                    st.caption("Sources: " + ", ".join(message["sources"]))

    if st.session_state.pending_summarize_question and not st.session_state.documents:
        st.session_state.pending_summarize_question = None

    if st.session_state.pending_summarize_question:
        with st.chat_message("assistant"):
            st.markdown("Which document would you like to summarize?")
            library = list(st.session_state.documents.items())
            valid_ids = {doc_id for doc_id, _ in library}
            if st.session_state.get("summarize-choice") not in valid_ids:
                st.session_state.pop("summarize-choice", None)
            chosen_id = st.radio(
                "Choose a document",
                options=[doc_id for doc_id, _ in library],
                format_func=lambda doc_id: st.session_state.documents[doc_id]["name"],
                index=None,
                key="summarize-choice",
                label_visibility="collapsed",
            )
            if st.button("Summarize", key="summarize-confirm", disabled=chosen_id is None):
                question = st.session_state.pending_summarize_question
                document_names = {doc_id: meta["name"] for doc_id, meta in st.session_state.documents.items()}
                history_for_prompt = [
                    (m["role"], m["text"]) for m in st.session_state.chat_history[:-1]
                ]
                with st.spinner("Summarizing…"):
                    message = build_answer_message(
                        get_vector_store(),
                        [chosen_id],
                        question,
                        document_names=document_names,
                        chat_history=history_for_prompt,
                    )
                    st.session_state.chat_history.append(message)
                st.session_state.pending_summarize_question = None
                st.rerun()

    documents_ready = bool(st.session_state.documents) and not st.session_state.pending_summarize_question
    question = st.chat_input(
        "Ask a question about your documents" if st.session_state.documents else "Upload a document to get started",
        disabled=not documents_ready,
    )
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question and documents_ready:
        st.session_state.chat_history.append({"role": "user", "text": question})
        document_ids = list(st.session_state.documents.keys())
        document_names = {doc_id: meta["name"] for doc_id, meta in st.session_state.documents.items()}

        if detect_question_type(question) == "summarization" and len(document_ids) > 1:
            st.session_state.pending_summarize_question = question
            st.rerun()
        else:
            history_for_prompt = [
                (m["role"], m["text"]) for m in st.session_state.chat_history[:-1]
            ]
            with st.spinner("Searching documents…"):
                message = build_answer_message(
                    get_vector_store(),
                    document_ids,
                    question,
                    document_names=document_names,
                    chat_history=history_for_prompt,
                )
                st.session_state.chat_history.append(message)
            st.rerun()
