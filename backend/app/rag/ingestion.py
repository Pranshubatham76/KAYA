"""
SentinelSite — RAG Ingestion Pipeline
PDF → unstructured → recursive chunking → OpenAI embeddings → Qdrant
One Qdrant collection per site, strict site_id namespace.
"""
from __future__ import annotations

import hashlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import settings
from app.core.storage import storage
from app.core.vector_db import vector_db
from app.db.models import DocumentStatus, DocumentType, SiteDocument

log = logging.getLogger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize extracted text — remove artefacts from PDF extraction."""
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse excessive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)           # collapse spaces
    text = re.sub(r"-\n(\w)", r"\1", text)            # rejoin hyphenated words across lines
    text = re.sub(r"(\w)\n(\w)", r"\1 \2", text)     # soft-wrap to single space
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = settings.RAG_CHUNK_SIZE,
    overlap: int = settings.RAG_CHUNK_OVERLAP,
) -> list[str]:
    """
    Recursive character-based chunking with overlap.
    Priority separators: paragraph → sentence → word.
    Rough token estimate: 1 token ≈ 4 chars.
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    chunks: list[str] = []

    def split(s: str, sep_idx: int) -> None:
        if len(s) <= char_size:
            if s.strip():
                chunks.append(s.strip())
            return
        if sep_idx >= len(separators):
            # Force-split
            for i in range(0, len(s), char_size - char_overlap):
                piece = s[i : i + char_size]
                if piece.strip():
                    chunks.append(piece.strip())
            return
        sep = separators[sep_idx]
        parts = s.split(sep) if sep else list(s)
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= char_size:
                current = candidate
            else:
                if current.strip():
                    split(current, sep_idx + 1)
                current = part
        if current.strip():
            split(current, sep_idx + 1)

    split(text, 0)

    # Apply overlap: prepend tail of previous chunk
    result: list[str] = []
    for i, chunk in enumerate(chunks):
        if i > 0 and char_overlap > 0:
            prev_tail = chunks[i - 1][-char_overlap:]
            chunk = prev_tail + " " + chunk
        result.append(chunk)

    return result


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """
    Extract text from PDF using unstructured.
    Returns list of {text, page_number, element_type}.
    Falls back to PyPDF2 if unstructured fails.
    """
    try:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",           # "hi_res" for scanned, "fast" for digital
            include_page_breaks=True,
        )
        pages: list[dict] = []
        current_page = 1
        page_text: list[str] = []

        for elem in elements:
            if hasattr(elem, "metadata") and elem.metadata.page_number:
                page_num = elem.metadata.page_number
            else:
                page_num = current_page

            if page_num != current_page and page_text:
                pages.append(
                    {
                        "text": _clean_text(" ".join(page_text)),
                        "page_number": current_page,
                        "element_type": "page",
                    }
                )
                page_text = []
                current_page = page_num

            text = str(elem).strip()
            if text:
                page_text.append(text)

        if page_text:
            pages.append(
                {
                    "text": _clean_text(" ".join(page_text)),
                    "page_number": current_page,
                    "element_type": "page",
                }
            )
        return [p for p in pages if p["text"]]

    except ImportError:
        log.warning("unstructured not installed — falling back to pypdf")
        return _extract_pdf_fallback(pdf_path)
    except Exception as e:
        log.error(f"PDF extraction failed: {e}")
        raise


def _extract_pdf_fallback(pdf_path: str) -> list[dict[str, Any]]:
    """PyPDF2 fallback for simple digital PDFs."""
    try:
        import pypdf

        pages = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(
                        {
                            "text": _clean_text(text),
                            "page_number": i + 1,
                            "element_type": "page",
                        }
                    )
        return pages
    except Exception as e:
        log.error(f"PDF fallback extraction failed: {e}")
        raise


# ── Point ID generation ───────────────────────────────────────────────────────

def _point_id(doc_id: str, chunk_index: int) -> str:
    """
    Stable, deterministic Qdrant point ID.
    SHA-256 of doc_id+chunk_index, truncated to UUID format.
    Qdrant accepts unsigned int64 or UUID string.
    """
    raw = f"{doc_id}:{chunk_index}"
    h = hashlib.sha256(raw.encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ── Main ingestion ────────────────────────────────────────────────────────────

async def ingest_document(
    site_id: str | UUID,
    doc_id: str,
    doc_type: DocumentType,
    filename: str,
    s3_key: str,
) -> dict[str, int]:
    """
    Full ingestion pipeline:
    1. Download PDF from S3
    2. Extract text by page
    3. Chunk with overlap
    4. Embed via OpenAI
    5. Upsert to Qdrant

    Returns {"n_chunks": N, "n_tokens": M}
    Called by Celery task.
    """
    log.info(f"Ingesting document: {filename} (doc_id={doc_id}, site={site_id})")

    # Step 1: Download
    pdf_bytes = storage.download_document(s3_key)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Step 2: Extract
    try:
        pages = _extract_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not pages:
        raise ValueError(f"No text extracted from {filename}")

    # Step 3: Chunk
    all_chunks: list[dict[str, Any]] = []
    chunk_index = 0
    total_tokens = 0

    for page in pages:
        page_chunks = _chunk_text(page["text"])
        for chunk_text in page_chunks:
            token_count = len(chunk_text) // 4  # rough estimate
            total_tokens += token_count
            all_chunks.append(
                {
                    "id": _point_id(doc_id, chunk_index),
                    "text": chunk_text,
                    "doc_id": doc_id,
                    "doc_type": doc_type.value,
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "page_number": page["page_number"],
                    "token_count": token_count,
                }
            )
            chunk_index += 1

    if not all_chunks:
        raise ValueError(f"Chunking produced 0 chunks for {filename}")

    log.info(f"Document chunked: {len(all_chunks)} chunks, ~{total_tokens} tokens")

    # Step 4 + 5: Embed + Upsert
    n_upserted = await vector_db.upsert_chunks(site_id, all_chunks)

    return {"n_chunks": n_upserted, "n_tokens": total_tokens}


async def delete_document(site_id: str | UUID, doc_id: str) -> None:
    """Remove all chunks for a document from Qdrant (on replacement/deletion)."""
    await vector_db.delete_document_chunks(site_id, doc_id)
    log.info(f"Removed document {doc_id} from Qdrant (site={site_id})")
