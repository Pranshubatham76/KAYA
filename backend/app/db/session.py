"""
SentinelSite — Database Session Management
Async (FastAPI routes) + Sync (Celery workers) engines in one place.
"""
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import settings
from app.db.models import Base

import logging

log = logging.getLogger(__name__)

# ── Async Engine (FastAPI) ────────────────────────────────────────────────────

async_engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,       # reconnect on stale connections
    pool_recycle=3600,        # recycle connections every 1h
    echo=settings.DEBUG,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,   # avoid lazy-load errors after commit
    autoflush=False,
    autocommit=False,
)


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency. Use in route:
        db: AsyncSession = Depends(get_async_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def async_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for use outside FastAPI (e.g. startup events, scripts).
        async with async_db_context() as db:
            result = await db.execute(...)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Sync Engine (Celery workers) ─────────────────────────────────────────────

sync_engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """
    Context manager for Celery tasks and scripts.
        with get_sync_db() as db:
            db.query(...)
    """
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Migrations helper ─────────────────────────────────────────────────────────

async def create_all_tables() -> None:
    """
    Used in tests / local dev only.
    Production uses Alembic migrations.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("All tables created via create_all (dev mode)")


async def drop_all_tables() -> None:
    """Nuke all tables — tests only."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    log.warning("All tables dropped")


# ── Health check ──────────────────────────────────────────────────────────────

async def check_db_health() -> dict:
    """Called by /health endpoint."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ok", "db": "postgres"}
    except Exception as e:
        log.error(f"DB health check failed: {e}")
        return {"status": "error", "db": "postgres", "detail": str(e)}
