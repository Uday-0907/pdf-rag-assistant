# app.py
"""Module 7: Streamlit interface for the Domain-Specific RAG Chatbot."""

import os

import streamlit as st
from dotenv import load_dotenv

# Local development: read GOOGLE_API_KEY from .env
load_dotenv()

# Streamlit Cloud: read GOOGLE_API_KEY from st.secrets (no .env file there).
try:
    if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass  # no secrets file configured - that is fine locally

from document_loader import MAX_FILE_MB, extract_text_from_pdfs  # noqa: E402
from rag_pipeline import MissingAPIKeyError, generate_answer  # noqa: E402
from vector_store import (  # noqa: E402
    create_vector_store,
    delete_saved_index,
    load_vector_store,
    save_vector_store,
    saved_index_exists,
)

st.set_page_config(page_title="Domain-Specific RAG Chatbot", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "vector_store": None,   # FAISS index for the current documents
        "indexed_files": [],    # [{"name", "pages", "empty_pages"}]
        "skipped_files": [],    # [{"name", "reason"}]
        "chat_history": [],     # [{"role", "content", "sources"}]
        "uploader_key": 0,      # changing this resets the file uploader widget
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


# ---------------------------------------------------------------------------
# Button callbacks
# ---------------------------------------------------------------------------
def clear_chat() -> None:
    st.session_state.chat_history = []


def clear_documents() -> None:
    """Remove the indexed documents (and the saved index on disk), the chat, and the uploads."""
    st.session_state.vector_store = None
    st.session_state.indexed_files = []
    st.session_state.skipped_files = []
    st.session_state.chat_history = []
    st.session_state.uploader_key += 1
    delete_saved_index()


def process_documents(uploaded_files, persist: bool) -> None:
    """Extract -> chunk -> embed -> index. Replaces any previously indexed documents."""
    with st.spinner("Extracting text, creating embeddings and building the FAISS index..."):
        documents, loaded, skipped = extract_text_from_pdfs(uploaded_files)
        st.session_state.skipped_files = skipped

        if not documents:
            st.error("None of the selected files could be used. See the reasons below.")
            return

        try:
            store = create_vector_store(documents)
        except Exception as exc:
            st.error(f"Could not build the vector store ({type(exc).__name__}): {exc}")
            return

        st.session_state.vector_store = store
        st.session_state.indexed_files = loaded
        st.session_state.chat_history = []  # new documents -> start a fresh conversation

        if persist:
            try:
                save_vector_store(store, loaded)
            except Exception as exc:
                st.warning(f"The documents were indexed but the index could not be saved: {exc}")

    st.success(f"Indexed {len(loaded)} PDF file(s). You can now ask questions.")


def load_saved_index() -> None:
    try:
        store, files = load_vector_store()
    except Exception as exc:
        st.error(f"Could not load the saved index ({type(exc).__name__}): {exc}")
        return
    if store is None:
        st.warning("No saved index was found.")
        return
    st.session_state.vector_store = store
    st.session_state.indexed_files = files
    st.session_state.skipped_files = []
    st.session_state.chat_history = []
    st.success("Saved index loaded.")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_sources(sources) -> None:
    """Section 3: display answer with source document and page."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            st.markdown(f"**{src['source']}** - page {src['page']}  |  relevance {src['score']:.0%}")
            st.caption(src["snippet"])


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"))


# ---------------------------------------------------------------------------
# Sidebar: document upload (Module 1) and controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Operations")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"PDF only, up to {MAX_FILE_MB} MB per file.",
    )
    st.caption(f"PDF only - max {MAX_FILE_MB} MB per file. Do not upload confidential documents without permission.")

    persist = st.checkbox(
        "Save index on this computer for reuse",
        value=False,
        help=(
            "Stores the index in vector_store/saved_index/. Leave OFF on shared or "
            "deployed servers, because the saved index contains text from your documents."
        ),
    )

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please choose at least one PDF file first.")
        else:
            process_documents(uploaded_files, persist)

    if saved_index_exists():
        st.button("Load saved index", on_click=load_saved_index)

    # Show which documents are active, and which were rejected.
    if st.session_state.indexed_files:
        st.subheader("Indexed documents")
        for info in st.session_state.indexed_files:
            details = f"{info.get('pages', '?')} pages"
            if info.get("empty_pages"):
                details += f", {info['empty_pages']} empty page(s) skipped"
            st.write(f"📄 **{info['name']}** ({details})")
    for item in st.session_state.skipped_files:
        st.warning(f"Skipped **{item['name']}**: {item['reason']}")

    st.divider()
    col_a, col_b = st.columns(2)
    col_a.button("Clear Chat", on_click=clear_chat)
    col_b.button(
        "Clear Documents",
        on_click=clear_documents,
        help="Removes the indexed documents, the chat and any saved index. Upload new files to replace them.",
    )


# ---------------------------------------------------------------------------
# Main page: chat
# ---------------------------------------------------------------------------
st.title("📄 Domain-Specific RAG Chatbot")
st.caption("Upload PDFs and ask questions. Answers are generated only from your documents.")
st.info(
    "AI-generated answers can be wrong or incomplete. Always verify high-stakes "
    "information (legal, medical, financial, HR or safety) in the original document."
)

if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "GOOGLE_API_KEY is not set. Add it to a .env file (local) or to Streamlit "
        "secrets (cloud), then restart the app."
    )

if st.session_state.vector_store is None:
    st.write("👈 Upload one or more PDFs in the sidebar and click **Process Documents** to begin.")

for message in st.session_state.chat_history:
    render_message(message)

query = st.chat_input("Ask a question about your documents...")

if query:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process at least one PDF file first.")
    else:
        user_message = {"role": "user", "content": query}
        render_message(user_message)

        result = None
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    result = generate_answer(
                        query,
                        st.session_state.vector_store,
                        chat_history=st.session_state.chat_history,
                    )
                except MissingAPIKeyError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Could not generate an answer ({type(exc).__name__}): {str(exc)[:300]}")

            if result is not None:
                st.markdown(result["answer"])
                render_sources(result["sources"])

        if result is not None:
            st.session_state.chat_history.append(user_message)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
