# vector_store.py
"""Module 3 (chunking) and Module 4 (embeddings + FAISS vector store)."""

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suggested beginner setting from the guidance: chunk 700-1000, overlap 100-150 characters.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# The guidance calls this model "all-MiniLM-L6-v2" (full Hugging Face id below).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Where the index is saved when the user chooses to keep it (see the sidebar checkbox).
INDEX_DIR = Path(__file__).resolve().parent / "vector_store" / "saved_index"
MANIFEST_NAME = "manifest.json"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per process (loading takes a few seconds).

    Embeddings are normalised to length 1, so distance in FAISS maps directly to
    cosine similarity (see rag_pipeline.retrieve).
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(documents: List[Document]) -> List[Document]:
    """Split page documents into overlapping chunks. Metadata (source, page) is kept."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def create_vector_store(documents: List[Document]) -> FAISS:
    """Chunk the documents, embed every chunk and build a FAISS index."""
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks could be created from the documents.")
    return FAISS.from_documents(chunks, get_embeddings())


# ---------------------------------------------------------------------------
# Optional persistence ("Save the vector store locally if the application must reuse it")
# ---------------------------------------------------------------------------

def saved_index_exists(path: Path = INDEX_DIR) -> bool:
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def save_vector_store(store: FAISS, files_info: List[Dict], path: Path = INDEX_DIR) -> None:
    """Save the FAISS index plus a small manifest listing the indexed files."""
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    manifest = {"files": files_info}
    (path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_vector_store(path: Path = INDEX_DIR) -> Tuple[Optional[FAISS], List[Dict]]:
    """Load a previously saved index. Returns (None, []) if nothing is saved."""
    if not saved_index_exists(path):
        return None, []
    # FAISS.load_local uses pickle. That is only safe for files YOU created, which is
    # the case here (the app writes this folder itself). Never load an index from
    # an untrusted source.
    store = FAISS.load_local(
        str(path), get_embeddings(), allow_dangerous_deserialization=True
    )
    files: List[Dict] = []
    manifest_file = path / MANIFEST_NAME
    if manifest_file.exists():
        try:
            files = json.loads(manifest_file.read_text(encoding="utf-8")).get("files", [])
        except (OSError, ValueError):
            files = []
    return store, files


def delete_saved_index(path: Path = INDEX_DIR) -> None:
    shutil.rmtree(path, ignore_errors=True)
