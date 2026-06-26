"""
SentinelSite — FastAPI Application Entry Point
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.config import settings
from app.db.session import check_db_health
from app.core.vector_db import vector_db
from app.core.storage import storage

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown lifecycle."""
    log.info(f"Starting {settings.APP_NAME} [{settings.APP_ENV}]")

    # Ensure S3 buckets exist (local dev / MinIO)
    if settings.USE_MINIO:
        try:
            storage.ensure_buckets_exist()
        except Exception as e:
            log.warning(f"S3 bucket creation failed (non-fatal): {e}")

    # Ensure Qdrant collections exist for all active sites (warm-up)
    # Real init happens lazily on first document upload

    log.info("SentinelSite started")
    yield
    log.info("SentinelSite shutting down")


app = FastAPI(
    title="SentinelSite API",
    version="1.0.0",
    description="Passive near-miss detection system for construction sites",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["infra"])
async def health():
    db_health = await check_db_health()
    qdrant_health = await vector_db.check_health()
    s3_health = storage.check_health()

    all_ok = all(
        h["status"] == "ok"
        for h in [db_health, qdrant_health, s3_health]
    )
    return {
        "status": "ok" if all_ok else "degraded",
        "components": {
            "db": db_health,
            "qdrant": qdrant_health,
            "storage": s3_health,
        },
        "version": "1.0.0",
        "env": settings.APP_ENV,
    }


# ── Routes (registered here — you add these) ──────────────────────────────────
from app.api import events, voice, training, models, documents, ws
app.include_router(events.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(ws.router)
