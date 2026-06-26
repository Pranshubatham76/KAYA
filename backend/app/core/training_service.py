"""
SentinelSite — Training Service
Service layer over Celery training tasks + DB.
Handles: manual trigger, job status, confidence score updates,
         admin few-shot training, training sample queries.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy import and_, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import (
    AdminTrainingImage, ModelStatus, ModelType,
    ModelVersion, TrainingJob, TrainingJobStatus,
    TrainingSample, User, UserRole,
)
from app.training.scheduler import (
    check_training_conditions,
    _enqueue_training_job,
)

log = logging.getLogger(__name__)

# Admin training gates (from master ref FR-A01)
ADMIN_MIN_IMAGES_PER_CLASS = 15
ADMIN_NEW_CLASS_ACC_THRESHOLD = 0.70
ADMIN_EXISTING_CLASS_DROP_LIMIT = 0.05


class TrainingService:

    # ── Manual training trigger ───────────────────────────────────────────────

    async def trigger_training(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: str,
        acting_user: User,
        replay_strategy: str = "class_balanced",
        force: bool = False,
    ) -> TrainingJob:
        """
        Manually trigger a training job (admin only).
        force=True bypasses the 3 scheduler conditions (used by admin).
        """
        from app.core.auth_service import auth_service
        auth_service.require_role(acting_user, UserRole.ADMIN, UserRole.SUPERVISOR)
        auth_service.require_site_access(acting_user, site_id)

        # Check conditions unless forced
        if not force:
            from app.db.session import get_sync_db
            with get_sync_db() as sync_db:
                should_trigger, debug = check_training_conditions(
                    sync_db, site_id, model_type
                )
            if not should_trigger:
                raise ValueError(
                    f"Training conditions not met: {debug}. "
                    "Use force=True to override (admin only)."
                )

        from app.db.session import get_sync_db
        with get_sync_db() as sync_db:
            job_id = _enqueue_training_job(
                db=sync_db,
                site_id=site_id,
                model_type=model_type,
                trigger="admin",
                replay_strategy=replay_strategy,
            )

        # Fetch the created job to return
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        log.info(f"Training manually triggered: job={job_id}, site={site_id}, type={model_type}")
        return job

    # ── Job status & history ──────────────────────────────────────────────────

    async def get_job(
        self,
        db: AsyncSession,
        job_id: str,
    ) -> TrainingJob | None:
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: str | None = None,
        status: TrainingJobStatus | None = None,
        limit: int = 20,
    ) -> list[TrainingJob]:
        filters = [TrainingJob.site_id == site_id]
        if model_type:
            filters.append(TrainingJob.model_type == ModelType(model_type))
        if status:
            filters.append(TrainingJob.status == status)

        result = await db.execute(
            select(TrainingJob)
            .where(and_(*filters))
            .order_by(desc(TrainingJob.queued_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_training_readiness(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> dict:
        """
        Check if site is ready for training — used by dashboard status widget.
        Returns condition status for both acoustic + visual.
        """
        from app.db.session import get_sync_db
        report = {}
        with get_sync_db() as sync_db:
            for mt in ["acoustic", "visual"]:
                should, debug = check_training_conditions(sync_db, site_id, mt)
                report[mt] = {
                    "ready": should,
                    **debug,
                }
        return report

    # ── Training samples ──────────────────────────────────────────────────────

    async def list_training_samples(
        self,
        db: AsyncSession,
        site_id: str,
        used: bool | None = None,
        label_acoustic: str | None = None,
        limit: int = 100,
    ) -> list[TrainingSample]:
        filters = [TrainingSample.site_id == site_id]
        if used is not None:
            filters.append(TrainingSample.is_used_in_training == used)
        if label_acoustic:
            filters.append(TrainingSample.label_acoustic == label_acoustic)

        result = await db.execute(
            select(TrainingSample)
            .where(and_(*filters))
            .order_by(desc(TrainingSample.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_sample_counts(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> dict:
        """
        Summary of training samples for dashboard.
        Returns counts by label and used/unused breakdown.
        """
        # Total by class (acoustic label)
        acoustic_counts = await db.execute(
            select(TrainingSample.label_acoustic, func.count())
            .where(TrainingSample.site_id == site_id)
            .group_by(TrainingSample.label_acoustic)
        )

        # Unused (pending training)
        unused_count_res = await db.execute(
            select(func.count()).select_from(TrainingSample).where(
                TrainingSample.site_id == site_id,
                TrainingSample.is_used_in_training == False,
            )
        )
        unused = unused_count_res.scalar() or 0

        total_res = await db.execute(
            select(func.count()).select_from(TrainingSample).where(
                TrainingSample.site_id == site_id
            )
        )
        total = total_res.scalar() or 0

        return {
            "total": total,
            "unused_new": unused,
            "min_required_for_trigger": settings.TRAINING_MIN_NEW_SAMPLES,
            "ready_to_trigger": unused >= settings.TRAINING_MIN_NEW_SAMPLES,
            "by_acoustic_class": {
                row[0] or "unlabeled": row[1]
                for row in acoustic_counts.all()
            },
        }

    async def update_sample_confidence(
        self,
        db: AsyncSession,
        site_id: str,
        confidences: dict[str, float],  # {sample_id: confidence}
    ) -> int:
        """
        Update last_model_confidence for training samples.
        Called after each model evaluation pass (uncertainty-weighted replay).
        Returns number of samples updated.
        """
        if not confidences:
            return 0

        now = datetime.utcnow()
        updated = 0
        for sample_id, confidence in confidences.items():
            result = await db.execute(
                select(TrainingSample).where(
                    TrainingSample.id == sample_id,
                    TrainingSample.site_id == site_id,
                )
            )
            sample = result.scalar_one_or_none()
            if sample:
                sample.last_model_confidence = confidence
                sample.last_evaluated_at = now
                updated += 1

        await db.commit()
        log.info(f"Updated confidence for {updated} samples (site={site_id})")
        return updated

    # ── Admin few-shot training ───────────────────────────────────────────────

    async def trigger_admin_training(
        self,
        db: AsyncSession,
        site_id: str,
        class_label: str,
        acting_user: User,
    ) -> TrainingJob:
        """
        Admin triggers few-shot visual training for a new class.
        Validates min 15 images exist, then enqueues job.
        """
        from app.core.auth_service import auth_service
        auth_service.require_role(acting_user, UserRole.ADMIN)
        auth_service.require_site_access(acting_user, site_id)

        # Count images for this class
        count_res = await db.execute(
            select(func.count()).select_from(AdminTrainingImage).where(
                AdminTrainingImage.site_id == site_id,
                AdminTrainingImage.class_label == class_label,
            )
        )
        n_images = count_res.scalar() or 0

        if n_images < ADMIN_MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Need at least {ADMIN_MIN_IMAGES_PER_CLASS} images for class "
                f"'{class_label}', have {n_images}."
            )

        from app.db.session import get_sync_db
        with get_sync_db() as sync_db:
            job_id = _enqueue_training_job(
                db=sync_db,
                site_id=site_id,
                model_type="visual",
                trigger="admin",
                replay_strategy="class_balanced",
            )

        result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one_or_none()
        log.info(f"Admin training triggered: class={class_label}, site={site_id}, job={job_id}")
        return job

    async def get_admin_training_status(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> list[dict]:
        """
        Per-class image counts + readiness for dashboard admin panel.
        """
        counts_res = await db.execute(
            select(AdminTrainingImage.class_label, func.count())
            .where(AdminTrainingImage.site_id == site_id)
            .group_by(AdminTrainingImage.class_label)
        )
        rows = counts_res.all()

        # Last training job per class (approximation — by visual model job time)
        last_job_res = await db.execute(
            select(TrainingJob)
            .where(
                TrainingJob.site_id == site_id,
                TrainingJob.model_type == ModelType.VISUAL,
                TrainingJob.status == TrainingJobStatus.COMPLETED,
            )
            .order_by(desc(TrainingJob.completed_at))
            .limit(1)
        )
        last_job = last_job_res.scalar_one_or_none()

        return [
            {
                "class_label": class_label,
                "n_images": n,
                "is_ready": n >= ADMIN_MIN_IMAGES_PER_CLASS,
                "last_trained_at": last_job.completed_at.isoformat() if last_job else None,
            }
            for class_label, n in rows
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────
training_service = TrainingService()
