"""
SentinelSite — RAG Retriever
Hybrid search (dense + BM25) → cross-encoder reranking → top-k chunks.
Returns ranked SourceChunks with scores and metadata.
"""
from __future__ import annotations

import logging
import time
from typing import Any, NamedTuple
from uuid import UUID

from app.config import settings
from app.core.vector_db import vector_db
from app.db.models import DocumentType

log = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

class RetrievedChunk(NamedTuple):
    point_id: str
    text: str
    doc_id: str
    doc_type: str
    filename: str
    chunk_index: int
    page_number: int | None
    score: float          # final score after reranking
    raw_score: float      # original vector similarity score


# ── Reranker ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.
    Fallback: pass-through with original scores.
    """

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            log.info("Cross-encoder reranker loaded")
        except ImportError:
            log.warning("sentence-transformers not installed — skipping reranking")
            self._model = "passthrough"

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Score (query, chunk) pairs and return top_k re-sorted.
        """
        self._load()
        if not chunks or self._model == "passthrough":
            return chunks[:top_k]

        pairs = [[query, c.text] for c in chunks]
        scores: list[float] = self._model.predict(pairs).tolist()

        reranked = [
            RetrievedChunk(
                point_id=c.point_id,
                text=c.text,
                doc_id=c.doc_id,
                doc_type=c.doc_type,
                filename=c.filename,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                score=float(scores[i]),
                raw_score=c.raw_score,
            )
            for i, c in enumerate(chunks)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Main retrieval class.
    retrieve() is the only method you call from outside.
    """

    def __init__(self) -> None:
        self._reranker = CrossEncoderReranker()

    async def retrieve(
        self,
        site_id: str | UUID,
        query: str,
        doc_type_filter: DocumentType | None = None,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> tuple[list[RetrievedChunk], dict[str, Any]]:
        """
        Full retrieval pipeline:
        1. Hybrid search (dense + BM25 RRF)
        2. Cross-encoder reranking
        3. Return top-k chunks

        Returns (chunks, debug_info)
        """
        fetch_k = (top_k or settings.RAG_TOP_K)
        rerank_top_k = settings.RAG_RERANK_TOP_K

        t0 = time.perf_counter()

        # Hybrid search
        doc_type_str = doc_type_filter.value if doc_type_filter else None
        raw_results = await vector_db.hybrid_search(
            site_id=site_id,
            query=query,
            top_k=fetch_k * 2,  # over-fetch for reranker
            doc_type_filter=doc_type_str,
        )

        t_search = time.perf_counter() - t0

        if not raw_results:
            return [], {
                "n_retrieved": 0,
                "n_reranked": 0,
                "search_ms": round(t_search * 1000),
                "rerank_ms": 0,
                "doc_type_filter": doc_type_str,
            }

        # Convert to RetrievedChunk
        chunks = [
            RetrievedChunk(
                point_id=str(hit.id),
                text=hit.payload.get("text", ""),
                doc_id=hit.payload.get("doc_id", ""),
                doc_type=hit.payload.get("doc_type", "general"),
                filename=hit.payload.get("filename", ""),
                chunk_index=hit.payload.get("chunk_index", 0),
                page_number=hit.payload.get("page_number"),
                score=hit.score,
                raw_score=hit.score,
            )
            for hit in raw_results
            if hit.payload.get("text")
        ]

        t_rerank_start = time.perf_counter()

        # Rerank
        if rerank and len(chunks) > 1:
            chunks = self._reranker.rerank(query, chunks, top_k=rerank_top_k)
        else:
            chunks = chunks[:rerank_top_k]

        t_rerank = time.perf_counter() - t_rerank_start

        debug = {
            "n_retrieved": len(raw_results),
            "n_reranked": len(chunks),
            "search_ms": round(t_search * 1000),
            "rerank_ms": round(t_rerank * 1000),
            "doc_type_filter": doc_type_str,
            "top_scores": [round(c.score, 4) for c in chunks],
        }

        log.debug(
            f"Retrieved {len(chunks)} chunks for site={site_id} "
            f"(search={debug['search_ms']}ms, rerank={debug['rerank_ms']}ms)"
        )
        return chunks, debug


# ── Singleton ─────────────────────────────────────────────────────────────────
retriever = Retriever()
