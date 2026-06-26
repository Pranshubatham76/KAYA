"""
SentinelSite — Qdrant Vector DB Client
One Qdrant collection per site: sentinel_{site_id}
Hybrid search (dense + sparse BM25) + cross-encoder reranking.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from app.config import settings

log = logging.getLogger(__name__)

# Sparse vector config for BM25-style keyword search
SPARSE_VECTOR_NAME = "text-sparse"
DENSE_VECTOR_NAME = "text-dense"


class VectorDBService:
    """
    Wraps AsyncQdrantClient.
    Collection naming: sentinel_{site_id}  (dashes stripped for Qdrant compat)
    """

    def __init__(self) -> None:
        client_kwargs: dict[str, Any] = {
            "host": settings.QDRANT_HOST,
            "port": settings.QDRANT_PORT,
            "prefer_grpc": True,
        }
        if settings.QDRANT_API_KEY:
            client_kwargs["api_key"] = settings.QDRANT_API_KEY

        self._client = AsyncQdrantClient(**client_kwargs)
        self._embed_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        log.info(f"VectorDB ready @ {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    # ── Collection helpers ────────────────────────────────────────────────────

    def collection_name(self, site_id: str | UUID) -> str:
        """sentinel_<site_id with dashes stripped>"""
        clean = str(site_id).replace("-", "")
        return f"{settings.QDRANT_COLLECTION_PREFIX}_{clean}"

    async def ensure_collection(self, site_id: str | UUID) -> str:
        """
        Create collection if it doesn't exist.
        Called on first document upload for a site.
        Returns collection name.
        """
        name = self.collection_name(site_id)
        try:
            await self._client.get_collection(name)
            log.debug(f"Collection already exists: {name}")
        except (UnexpectedResponse, Exception):
            await self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=settings.QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE,
                        on_disk=settings.QDRANT_ON_DISK,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=100,  # build HNSW after 100 vectors
                ),
            )
            # Payload index for fast metadata filtering
            await self._client.create_payload_index(
                collection_name=name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self._client.create_payload_index(
                collection_name=name,
                field_name="doc_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            await self._client.create_payload_index(
                collection_name=name,
                field_name="chunk_index",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
            log.info(f"Created Qdrant collection: {name}")
        return name

    async def delete_collection(self, site_id: str | UUID) -> None:
        name = self.collection_name(site_id)
        await self._client.delete_collection(name)
        log.info(f"Deleted Qdrant collection: {name}")

    async def collection_exists(self, site_id: str | UUID) -> bool:
        try:
            await self._client.get_collection(self.collection_name(site_id))
            return True
        except Exception:
            return False

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Batch embed via OpenAI text-embedding-3-small.
        Max 2048 texts per call — batch if larger.
        """
        if not texts:
            return []
        response = await self._embed_client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_text(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    # ── Upsert (ingestion) ────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        site_id: str | UUID,
        chunks: list[dict[str, Any]],
    ) -> int:
        """
        Upsert document chunks into Qdrant.
        Each chunk dict must have:
            id: str (qdrant point id, usually doc_id + chunk_index)
            text: str
            doc_id: str
            doc_type: str
            filename: str
            chunk_index: int
            page_number: int | None
            token_count: int
        Returns number of upserted points.
        """
        if not chunks:
            return 0

        collection_name = await self.ensure_collection(site_id)

        texts = [c["text"] for c in chunks]
        embeddings = await self.embed_texts(texts)

        points = [
            PointStruct(
                id=chunk["id"],
                vector={DENSE_VECTOR_NAME: embedding},
                payload={
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "doc_type": chunk["doc_type"],
                    "filename": chunk["filename"],
                    "chunk_index": chunk["chunk_index"],
                    "page_number": chunk.get("page_number"),
                    "token_count": chunk.get("token_count", 0),
                    "site_id": str(site_id),
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Batch upsert in groups of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            await self._client.upsert(
                collection_name=collection_name,
                points=points[i : i + batch_size],
                wait=True,
            )
        log.info(f"Upserted {len(points)} chunks into {collection_name}")
        return len(points)

    async def delete_document_chunks(
        self,
        site_id: str | UUID,
        doc_id: str,
    ) -> None:
        """Delete all chunks belonging to a document (on replacement)."""
        collection_name = self.collection_name(site_id)
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )
        log.info(f"Deleted chunks for doc_id={doc_id} from {collection_name}")

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        site_id: str | UUID,
        query: str,
        top_k: int | None = None,
        doc_type_filter: str | None = None,
        score_threshold: float | None = None,
    ) -> list[ScoredPoint]:
        """
        Dense vector search with optional doc_type filter.
        Returns top-k scored points above threshold.
        """
        k = top_k or settings.RAG_TOP_K
        threshold = score_threshold or settings.RAG_SCORE_THRESHOLD

        collection_name = self.collection_name(site_id)
        query_vector = await self.embed_text(query)

        search_filter = None
        if doc_type_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=doc_type_filter),
                    )
                ]
            )

        results = await self._client.search(
            collection_name=collection_name,
            query_vector=(DENSE_VECTOR_NAME, query_vector),
            query_filter=search_filter,
            limit=k,
            score_threshold=threshold,
            with_payload=True,
        )
        return results

    async def hybrid_search(
        self,
        site_id: str | UUID,
        query: str,
        top_k: int | None = None,
        doc_type_filter: str | None = None,
    ) -> list[ScoredPoint]:
        """
        Hybrid search: dense cosine + sparse BM25, fused via RRF.
        Falls back to dense-only if sparse index not available.
        """
        k = top_k or settings.RAG_TOP_K
        collection_name = self.collection_name(site_id)

        query_vector = await self.embed_text(query)
        sparse_vector = self._bm25_sparse(query)

        search_filter = None
        if doc_type_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=doc_type_filter),
                    )
                ]
            )

        try:
            results = await self._client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=k * 2,
                        filter=search_filter,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(**sparse_vector),
                        using=SPARSE_VECTOR_NAME,
                        limit=k * 2,
                        filter=search_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
            return results.points
        except Exception as e:
            log.warning(f"Hybrid search failed ({e}), falling back to dense")
            return await self.search(site_id, query, top_k=k, doc_type_filter=doc_type_filter)

    def _bm25_sparse(self, text: str) -> dict:
        """
        Minimal TF-IDF sparse vector for keyword matching.
        Production: replace with SPLADE or FastEmbed sparse encoder.
        """
        import math
        tokens = text.lower().split()
        tf: dict[int, float] = {}
        for token in tokens:
            idx = abs(hash(token)) % 30000  # map to fixed-size vocab space
            tf[idx] = tf.get(idx, 0) + 1
        # Normalize
        total = sum(tf.values()) or 1
        indices = list(tf.keys())
        values = [v / total for v in tf.values()]
        return {"indices": indices, "values": values}

    # ── Collection stats ──────────────────────────────────────────────────────

    async def collection_stats(self, site_id: str | UUID) -> dict:
        try:
            name = self.collection_name(site_id)
            info = await self._client.get_collection(name)
            return {
                "collection": name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Health ────────────────────────────────────────────────────────────────

    async def check_health(self) -> dict:
        try:
            await self._client.get_collections()
            return {"status": "ok", "qdrant": "connected"}
        except Exception as e:
            log.error(f"Qdrant health check failed: {e}")
            return {"status": "error", "qdrant": str(e)}


# ── Module-level singleton ────────────────────────────────────────────────────
vector_db = VectorDBService()
