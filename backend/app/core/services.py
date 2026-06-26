"""
SentinelSite — Core Services
Business logic layer. Routes call these, never raw DB.
  - EventService: near-miss event lifecycle
  - DocumentService: document upload + ingestion trigger
  - VisionDescriber: GPT-4o Vision for frame description
  - SiteService: site management
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, update
from sqlalchemy.orm import selectinload

from app.config import settings
from app.core.storage import storage
from app.db.models import (
    AdminTrainingImage, DocumentStatus, DocumentType, EventStatus,
    ModelType, NearMissEvent, OshaCategory, Site, SiteDocument,
    TrainingSample, User, UserRole, VoiceQueryLog,
)
from app.rag.llm_chain import answer_query
from app.schemas import NearMissEventIngest, NearMissEventReview, VoiceQueryRequest

log = logging.getLogger(__name__)


# ── Event Service ─────────────────────────────────────────────────────────────

class EventService:

    async def ingest_event(
        self,
        db: AsyncSession,
        payload: NearMissEventIngest,
        audio_bytes: bytes | None,
        frame_bytes: bytes | None,
    ) -> NearMissEvent:
        """
        Create NearMissEvent record + upload audio/frame to S3.
        Enqueues GPT-4o Vision description as background task.
        """
        event_id = str(uuid4())

        # Upload audio + frame to S3
        audio_key = None
        frame_key = None

        if audio_bytes:
            audio_key = storage.upload_audio(
                str(payload.site_id), event_id, audio_bytes
            )

        if frame_bytes:
            frame_key = storage.upload_frame(
                str(payload.site_id), event_id, frame_bytes
            )

        # Resolve worker User row (by worker_id hash)
        worker_result = await db.execute(
            select(User).where(
                User.worker_id == payload.worker_id,
                User.site_id == str(payload.site_id),
            )
        )
        worker = worker_result.scalar_one_or_none()

        event = NearMissEvent(
            id=event_id,
            site_id=str(payload.site_id),
            worker_id=str(worker.id) if worker else None,
            device_id=payload.device_id,
            event_timestamp=payload.event_timestamp,
            gps_lat=payload.gps_lat,
            gps_lon=payload.gps_lon,
            yamnet_class=payload.yamnet_class,
            yamnet_class_id=payload.yamnet_class_id,
            yamnet_confidence=payload.yamnet_confidence,
            anomaly_score=payload.anomaly_score,
            imu_jerk_magnitude=payload.imu_jerk_magnitude,
            imu_timestamp_delta_ms=payload.imu_timestamp_delta_ms,
            visual_class=payload.visual_class,
            visual_confidence=payload.visual_confidence,
            audio_s3_key=audio_key,
            frame_s3_key=frame_key,
            status=EventStatus.PENDING,
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)

        log.info(
            f"Event ingested: id={event_id}, site={payload.site_id}, "
            f"yamnet={payload.yamnet_class}, anomaly={payload.anomaly_score}"
        )

        # Enqueue async tasks
        if frame_key:
            self._enqueue_vision_description(event_id, str(payload.site_id), frame_key)
        self._enqueue_server_recheck(event_id, audio_key)

        return event

    async def review_event(
        self,
        db: AsyncSession,
        event_id: str,
        reviewer_id: str,
        review: NearMissEventReview,
    ) -> NearMissEvent:
        """
        Supervisor confirms or dismisses event.
        On CONFIRM → creates TrainingSample automatically.
        """
        result = await db.execute(
            select(NearMissEvent).where(NearMissEvent.id == event_id)
        )
        event = result.scalar_one_or_none()
        if not event:
            raise ValueError(f"Event {event_id} not found")
        if event.status != EventStatus.PENDING:
            raise ValueError(f"Event already reviewed: {event.status}")

        event.status = review.status
        event.reviewed_by = reviewer_id
        event.reviewed_at = datetime.utcnow()
        event.osha_category = review.osha_category
        event.severity = review.severity
        event.review_notes = review.review_notes

        if review.status == EventStatus.CONFIRMED:
            sample = TrainingSample(
                id=str(uuid4()),
                site_id=event.site_id,
                event_id=event.id,
                label_acoustic=event.yamnet_class,
                label_visual=event.visual_class,
                label_osha=event.osha_category,
                label_severity=event.severity,
                audio_s3_key=event.audio_s3_key,
                frame_s3_key=event.frame_s3_key,
                is_used_in_training=False,
            )
            db.add(sample)
            log.info(f"TrainingSample created for event {event_id}")

        await db.commit()
        await db.refresh(event)
        return event

    async def list_events(
        self,
        db: AsyncSession,
        site_id: str,
        status: EventStatus | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[NearMissEvent], int]:
        filters = [NearMissEvent.site_id == site_id]
        if status:
            filters.append(NearMissEvent.status == status)

        count_result = await db.execute(
            select(func.count()).select_from(NearMissEvent).where(and_(*filters))
        )
        total = count_result.scalar()

        result = await db.execute(
            select(NearMissEvent)
            .where(and_(*filters))
            .order_by(NearMissEvent.event_timestamp.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        events = result.scalars().all()
        return list(events), total

    async def get_event_with_urls(
        self,
        db: AsyncSession,
        event_id: str,
    ) -> dict:
        """Fetch event and generate presigned S3 URLs for audio/frame."""
        result = await db.execute(
            select(NearMissEvent).where(NearMissEvent.id == event_id)
        )
        event = result.scalar_one_or_none()
        if not event:
            raise ValueError(f"Event {event_id} not found")

        audio_url = storage.presigned_audio_url(event.audio_s3_key) if event.audio_s3_key else None
        frame_url = storage.presigned_frame_url(event.frame_s3_key) if event.frame_s3_key else None

        return {"event": event, "audio_url": audio_url, "frame_url": frame_url}

    async def get_event_by_id(
        self,
        db: AsyncSession,
        event_id: str,
        site_id: str,
    ) -> NearMissEvent | None:
        """Simple fetch by ID + site_id (no presigned URLs)."""
        result = await db.execute(
            select(NearMissEvent).where(
                NearMissEvent.id == event_id,
                NearMissEvent.site_id == site_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_events_for_heatmap(
        self,
        db: AsyncSession,
        site_id: str,
        days: int = 30,
        status: EventStatus = EventStatus.CONFIRMED,
    ) -> list[dict]:
        """
        Lightweight GPS + metadata fetch for Leaflet heatmap.
        Returns only fields needed for map pins — no audio/frame URLs.
        """
        from datetime import timedelta
        since = datetime.utcnow() - timedelta(days=days)
        result = await db.execute(
            select(
                NearMissEvent.id,
                NearMissEvent.gps_lat,
                NearMissEvent.gps_lon,
                NearMissEvent.event_timestamp,
                NearMissEvent.yamnet_class,
                NearMissEvent.anomaly_score,
                NearMissEvent.osha_category,
                NearMissEvent.severity,
            )
            .where(
                NearMissEvent.site_id == site_id,
                NearMissEvent.status == status,
                NearMissEvent.event_timestamp >= since,
                NearMissEvent.gps_lat.isnot(None),
                NearMissEvent.gps_lon.isnot(None),
            )
            .order_by(NearMissEvent.event_timestamp.desc())
        )
        return [
            {
                "event_id": str(r.id),
                "lat": r.gps_lat,
                "lon": r.gps_lon,
                "timestamp": r.event_timestamp.isoformat(),
                "yamnet_class": r.yamnet_class,
                "anomaly_score": r.anomaly_score,
                "osha_category": r.osha_category.value if r.osha_category else None,
                "severity": r.severity.value if r.severity else None,
            }
            for r in result.all()
        ]

    async def get_pending_count(self, db: AsyncSession, site_id: str) -> int:
        """Fast count of unreviewed events — for dashboard badge."""
        from sqlalchemy import func
        result = await db.execute(
            select(func.count()).select_from(NearMissEvent).where(
                NearMissEvent.site_id == site_id,
                NearMissEvent.status == EventStatus.PENDING,
            )
        )
        return result.scalar() or 0

    def _enqueue_vision_description(self, event_id: str, site_id: str, frame_key: str) -> None:
        """Queue GPT-4o Vision description as Celery task."""
        try:
            from app.training.scheduler import celery_app
            celery_app.send_task(
                "app.training.scheduler.recheck_event_task",
                kwargs={"event_id": event_id, "site_id": site_id, "frame_key": frame_key},
                queue="cpu",
            )
        except Exception as e:
            log.warning(f"Failed to enqueue vision description: {e}")

    def _enqueue_server_recheck(self, event_id: str, audio_key: str | None) -> None:
        """Queue server-side YAMNet recheck."""
        if not audio_key:
            return
        try:
            from app.training.scheduler import celery_app
            celery_app.send_task(
                "app.ml.yamnet_recheck_task",
                kwargs={"event_id": event_id, "audio_key": audio_key},
                queue="cpu",
            )
        except Exception as e:
            log.warning(f"Failed to enqueue YAMNet recheck: {e}")


# ── Document Service ──────────────────────────────────────────────────────────

class DocumentService:

    async def upload_document(
        self,
        db: AsyncSession,
        site_id: str,
        doc_type: DocumentType,
        filename: str,
        file_bytes: bytes,
        uploaded_by: str | None = None,
    ) -> SiteDocument:
        """Upload document to S3 and queue ingestion task."""
        doc_id = str(uuid4())
        s3_key = storage.upload_document(site_id, filename, file_bytes)

        doc = SiteDocument(
            id=doc_id,
            site_id=site_id,
            uploaded_by=uploaded_by,
            filename=s3_key.split("/")[-1],
            original_filename=filename,
            doc_type=doc_type,
            status=DocumentStatus.QUEUED,
            s3_key=s3_key,
            file_size_bytes=len(file_bytes),
            is_active=True,
        )
        db.add(doc)
        await db.commit()

        # Kick off Celery ingestion task
        try:
            from app.training.scheduler import ingest_document_task
            ingest_document_task.apply_async(
                kwargs={"doc_id": doc_id, "site_id": site_id},
                queue="cpu",
            )
        except Exception as e:
            log.warning(f"Failed to enqueue document ingestion: {e}")

        log.info(f"Document uploaded: {filename} → {s3_key}")
        return doc

    async def list_documents(
        self,
        db: AsyncSession,
        site_id: str,
        doc_type: DocumentType | None = None,
        active_only: bool = True,
    ) -> list[SiteDocument]:
        filters = [SiteDocument.site_id == site_id]
        if doc_type:
            filters.append(SiteDocument.doc_type == doc_type)
        if active_only:
            filters.append(SiteDocument.is_active == True)

        result = await db.execute(
            select(SiteDocument)
            .where(and_(*filters))
            .order_by(SiteDocument.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_document(
        self,
        db: AsyncSession,
        doc_id: str,
        site_id: str,
    ) -> SiteDocument | None:
        result = await db.execute(
            select(SiteDocument).where(
                SiteDocument.id == doc_id,
                SiteDocument.site_id == site_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_document_status(
        self,
        db: AsyncSession,
        doc_id: str,
        site_id: str,
    ) -> dict:
        """
        Poll ingestion status. Dashboard calls this while showing upload progress.
        """
        doc = await self.get_document(db, doc_id, site_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        return {
            "doc_id": str(doc.id),
            "filename": doc.original_filename,
            "status": doc.status.value,
            "n_chunks": doc.n_chunks,
            "n_tokens_total": doc.n_tokens_total,
            "indexed_at": doc.indexed_at.isoformat() if doc.indexed_at else None,
            "error": doc.ingestion_error,
        }

    async def retry_ingestion(
        self,
        db: AsyncSession,
        doc_id: str,
        site_id: str,
    ) -> None:
        """Re-queue a FAILED document for ingestion."""
        from app.db.models import DocumentStatus
        doc = await self.get_document(db, doc_id, site_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        if doc.status != DocumentStatus.FAILED:
            raise ValueError(f"Document is not in FAILED state (current: {doc.status.value})")
        doc.status = DocumentStatus.QUEUED
        doc.ingestion_error = None
        await db.commit()
        try:
            from app.training.scheduler import ingest_document_task
            ingest_document_task.apply_async(
                kwargs={"doc_id": doc_id, "site_id": site_id},
                queue="cpu",
            )
        except Exception as e:
            log.warning(f"Failed to re-queue ingestion: {e}")

    async def delete_document(
        self,
        db: AsyncSession,
        doc_id: str,
        site_id: str,
    ) -> None:
        """Soft delete + remove from Qdrant."""
        result = await db.execute(
            select(SiteDocument).where(
                SiteDocument.id == doc_id,
                SiteDocument.site_id == site_id,
            )
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        doc.is_active = False
        await db.commit()

        # Remove from Qdrant async
        try:
            from app.core.vector_db import vector_db
            import asyncio
            asyncio.create_task(vector_db.delete_document_chunks(site_id, doc_id))
        except Exception as e:
            log.warning(f"Qdrant chunk deletion failed: {e}")


# ── Voice / RAG Service ───────────────────────────────────────────────────────

class VoiceService:

    async def handle_query(
        self,
        db: AsyncSession,
        request: VoiceQueryRequest,
    ) -> dict:
        """
        End-to-end voice query:
        1. Route to RAG pipeline
        2. Optionally log (worker opt-in)
        3. Return answer + sources
        """
        from app.rag.llm_chain import answer_query
        import time

        t_start = time.perf_counter()
        rag_answer = await answer_query(
            site_id=request.site_id,
            query=request.query_text,
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000)

        # Log if opted in
        if request.log_query and request.worker_id:
            log_entry = VoiceQueryLog(
                id=str(uuid4()),
                site_id=str(request.site_id),
                query_text=request.query_text,
                detected_intent=rag_answer.intent,
                answer_text=rag_answer.answer,
                source_document_ids=[c.doc_id for c in rag_answer.chunks],
                latency_ms=latency_ms,
                was_answered=rag_answer.was_answered,
            )
            db.add(log_entry)
            await db.commit()

        return rag_answer.to_dict()


# ── Vision Describer (GPT-4o) ─────────────────────────────────────────────────

class VisionDescriber:

    PROMPT = (
        "You are analyzing a construction site safety image captured at the moment of a near-miss incident.\n"
        "Describe in 2-3 sentences:\n"
        "1. What is visible in the scene (workers, equipment, structures)\n"
        "2. What potential hazard or near-miss is evident\n"
        "3. Any safety violations visible (missing PPE, unsecured loads, etc.)\n"
        "Be factual and concise. Do not speculate beyond what is visible."
    )

    async def describe(self, frame_bytes: bytes) -> str:
        """Call GPT-4o Vision to describe a near-miss frame."""
        if not settings.OPENAI_API_KEY:
            return "Frame description unavailable (no OpenAI key configured)"

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            b64_image = base64.b64encode(frame_bytes).decode("utf-8")
            response = await client.chat.completions.create(
                model=settings.OPENAI_VISION_MODEL,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "low",  # faster + cheaper
                            },
                        },
                    ],
                }],
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"GPT-4o Vision failed: {e}")
            return f"Frame description unavailable: {str(e)[:100]}"


# ── Site Service ──────────────────────────────────────────────────────────────

class SiteService:

    async def create_site(
        self,
        db: AsyncSession,
        name: str,
        location: str | None = None,
        timezone: str = "UTC",
    ) -> Site:
        """Create a new construction site."""
        from uuid import uuid4
        site = Site(
            id=str(uuid4()),
            name=name,
            location=location,
            timezone=timezone,
            is_active=True,
        )
        db.add(site)
        await db.commit()
        await db.refresh(site)
        log.info(f"Site created: {name} (id={site.id})")
        return site

    async def list_sites(
        self,
        db: AsyncSession,
        active_only: bool = True,
    ) -> list[Site]:
        filters = []
        if active_only:
            filters.append(Site.is_active == True)
        from sqlalchemy import and_
        result = await db.execute(
            select(Site).where(and_(*filters) if filters else True).order_by(Site.name)
        )
        return list(result.scalars().all())

    async def get_site(self, db: AsyncSession, site_id: str) -> Site | None:
        result = await db.execute(select(Site).where(Site.id == site_id))
        return result.scalar_one_or_none()

    async def create_zone(
        self,
        db: AsyncSession,
        site_id: str,
        name: str,
        risk_weight: float = 1.0,
        polygon_geojson: dict | None = None,
    ) -> "SiteZone":
        """Create a named GPS zone within a site."""
        from uuid import uuid4
        from app.db.models import SiteZone
        zone = SiteZone(
            id=str(uuid4()),
            site_id=site_id,
            name=name,
            risk_weight=risk_weight,
            polygon_geojson=polygon_geojson,
        )
        db.add(zone)
        await db.commit()
        await db.refresh(zone)
        return zone

    async def list_zones(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> list["SiteZone"]:
        from app.db.models import SiteZone
        result = await db.execute(
            select(SiteZone).where(SiteZone.site_id == site_id).order_by(SiteZone.name)
        )
        return list(result.scalars().all())

    async def deactivate_site(self, db: AsyncSession, site_id: str) -> None:
        site = await self.get_site(db, site_id)
        if not site:
            raise ValueError(f"Site {site_id} not found")
        site.is_active = False
        await db.commit()

    async def update_thresholds(
        self,
        db: AsyncSession,
        site_id: str,
        acoustic_threshold: float | None,
        imu_threshold: float | None,
    ) -> Site:
        site = await self.get_site(db, site_id)
        if not site:
            raise ValueError(f"Site {site_id} not found")
        if acoustic_threshold is not None:
            site.acoustic_threshold = acoustic_threshold
        if imu_threshold is not None:
            site.imu_threshold = imu_threshold
        await db.commit()
        await db.refresh(site)
        return site


# ── Singletons ────────────────────────────────────────────────────────────────
event_service = EventService()
document_service = DocumentService()
voice_service = VoiceService()
vision_describer = VisionDescriber()
site_service = SiteService()
