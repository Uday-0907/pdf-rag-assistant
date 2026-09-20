# document_loader.py
"""Module 1 (upload validation) and Module 2 (text extraction) of the project guidance."""

import io
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document
from pypdf import PdfReader

# Section 12: limit uploaded file types and sizes.
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024


class PDFLoadError(Exception):
    """A file could not be used. The message is safe to show to the user."""


def _clean_text(text: str) -> str:
    """Light clean-up so chunks are not full of blank space."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_bytes(filename: str, data: bytes) -> Tuple[List[Document], Dict]:
    """Validate one PDF and return (page_documents, info).

    Each returned Document holds the text of ONE page and the metadata
    {"source": <file name>, "page": <1-based page number>}.
    Pages with no extractable text are skipped. Raises PDFLoadError if the file
    is unusable (wrong type, too large, corrupt, password-protected, or scanned).
    """
    if not filename.lower().endswith(".pdf"):
        raise PDFLoadError("not a .pdf file")
    if not data:
        raise PDFLoadError("the file is empty")
    if len(data) > MAX_FILE_BYTES:
        raise PDFLoadError(f"larger than the {MAX_FILE_MB} MB limit")
    # Check the real content, not just the extension.
    if b"%PDF-" not in data[:1024]:
        raise PDFLoadError("does not look like a real PDF file")

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PDFLoadError("password-protected PDFs are not supported")
        total_pages = len(reader.pages)
    except PDFLoadError:
        raise
    except Exception as exc:  # corrupt file, unsupported encryption, ...
        raise PDFLoadError(f"could not be read as a PDF ({type(exc).__name__})") from exc

    documents: List[Document] = []
    empty_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = _clean_text(page.extract_text() or "")
        except Exception:
            text = ""
        if not text:  # skip empty pages safely
            empty_pages += 1
            continue
        documents.append(
            Document(page_content=text, metadata={"source": filename, "page": page_number})
        )

    if not documents:
        raise PDFLoadError("no extractable text (scanned PDF? OCR is not supported)")

    return documents, {"name": filename, "pages": total_pages, "empty_pages": empty_pages}


def extract_text_from_pdfs(uploaded_files) -> Tuple[List[Document], List[Dict], List[Dict]]:
    """Process Streamlit uploads.

    Returns (documents, loaded, skipped):
      documents - one Document per non-empty page, across all files
      loaded    - [{"name", "pages", "empty_pages"}, ...] for files that worked
      skipped   - [{"name", "reason"}, ...] for files that were rejected
    """
    documents: List[Document] = []
    loaded: List[Dict] = []
    skipped: List[Dict] = []
    seen = set()

    for uploaded in uploaded_files:
        name = Path(uploaded.name).name  # drop any path components
        if name in seen:
            skipped.append({"name": name, "reason": "duplicate file name"})
            continue
        seen.add(name)

        try:
            pages, info = load_pdf_bytes(name, uploaded.getvalue())
        except PDFLoadError as exc:
            skipped.append({"name": name, "reason": str(exc)})
            continue

        documents.extend(pages)
        loaded.append(info)

    return documents, loaded, skipped
