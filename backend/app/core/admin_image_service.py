"""
SentinelSite — Admin Image Service
Manages images uploaded by admin for visual model few-shot fine-tuning.
Enforces min 15 images per class (FR-A01) before allowing training trigger.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.storage import storage
from app.db.models import (
    AdminTrainingImage, ModelType, TrainingJob,
    TrainingJobStatus, User, UserRole,
)

log = logging.getLogger(__name__)

MIN_IMAGES_PER_CLASS = 15   # FR-A01 gate


class AdminImageService:

    # ── Upload ────────────────────────────────────────────────────────────────

    async def upload_image(
        self,
        db: AsyncSession,
        site_id: str,
        class_label: str,
        filename: str,
        image_bytes: bytes,
        acting_user: User,
    ) -> AdminTrainingImage:
        """
        Upload a training image for a new visual class.
        Admin only. Image stored in S3 under site_id/class_label/ namespace.
        """
        from app.core.auth_service import auth_service
        auth_service.require_role(acting_user, UserRole.ADMIN, UserRole.SUPERVISOR)
        auth_service.require_site_access(acting_user, site_id)

        if not class_label or len(class_label.strip()) < 2:
            raise ValueError("class_label must be at least 2 characters")
        if len(image_bytes) == 0:
            raise ValueError("Empty image file")
        if len(image_bytes) > 20 * 1024 * 1024:  # 20MB
            raise ValueError("Image too large (max 20MB)")

        # Validate it's actually an image
        self._validate_image(image_bytes)

        class_label = class_label.strip().lower().replace(" ", "_")
        s3_key = storage.upload_admin_image(
            site_id=site_id,
            class_label=class_label,
            filename=filename,
            data=image_bytes,
        )

        record = AdminTrainingImage(
            id=str(uuid4()),
            site_id=site_id,
            uploaded_by=str(acting_user.id),
            s3_key=s3_key,
            filename=filename,
            class_label=class_label,
            file_size_bytes=len(image_bytes),
            is_used_in_training=False,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)

        count = await self._count_class_images(db, site_id, class_label)
        log.info(
            f"Admin image uploaded: class={class_label}, site={site_id}, "
            f"count={count}/{MIN_IMAGES_PER_CLASS}"
        )
        return record

    async def upload_batch(
        self,
        db: AsyncSession,
        site_id: str,
        class_label: str,
        files: list[tuple[str, bytes]],  # [(filename, bytes), ...]
        acting_user: User,
    ) -> list[AdminTrainingImage]:
        """Upload multiple images for the same class at once."""
        records = []
        for filename, img_bytes in files:
            record = await self.upload_image(
                db, site_id, class_label, filename, img_bytes, acting_user
            )
            records.append(record)
        return records

    # ── Query ─────────────────────────────────────────────────────────────────

    async def list_images(
        self,
        db: AsyncSession,
        site_id: str,
        class_label: str | None = None,
    ) -> list[AdminTrainingImage]:
        filters = [AdminTrainingImage.site_id == site_id]
        if class_label:
            filters.append(
                AdminTrainingImage.class_label == class_label.strip().lower().replace(" ", "_")
            )
        result = await db.execute(
            select(AdminTrainingImage)
            .where(and_(*filters))
            .order_by(AdminTrainingImage.class_label, AdminTrainingImage.created_at)
        )
        return list(result.scalars().all())

    async def get_class_summary(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> list[dict]:
        """
        Per-class image counts + readiness status.
        Used by dashboard admin panel (ImageUploader.tsx, TrainingStatus.tsx).
        """
        result = await db.execute(
            select(
                AdminTrainingImage.class_label,
                func.count().label("n_images"),
                func.max(AdminTrainingImage.created_at).label("last_uploaded"),
            )
            .where(AdminTrainingImage.site_id == site_id)
            .group_by(AdminTrainingImage.class_label)
            .order_by(AdminTrainingImage.class_label)
        )
        rows = result.all()

        # Find last completed visual training job
        last_job_res = await db.execute(
            select(TrainingJob)
            .where(
                TrainingJob.site_id == site_id,
                TrainingJob.model_type == ModelType.VISUAL,
                TrainingJob.status == TrainingJobStatus.COMPLETED,
                TrainingJob.trigger == "admin",
            )
            .order_by(desc(TrainingJob.completed_at))
            .limit(1)
        )
        last_job = last_job_res.scalar_one_or_none()

        return [
            {
                "class_label": row.class_label,
                "n_images": row.n_images,
                "is_ready": row.n_images >= MIN_IMAGES_PER_CLASS,
                "shortfall": max(0, MIN_IMAGES_PER_CLASS - row.n_images),
                "last_uploaded_at": row.last_uploaded.isoformat() if row.last_uploaded else None,
                "last_trained_at": (
                    last_job.completed_at.isoformat() if last_job and last_job.completed_at else None
                ),
                "has_new_since_training": (
                    row.last_uploaded > last_job.completed_at
                    if last_job and last_job.completed_at and row.last_uploaded else True
                ),
            }
            for row in rows
        ]

    async def get_image_records_for_training(
        self,
        db: AsyncSession,
        site_id: str,
        class_label: str | None = None,
    ) -> list[dict]:
        """
        Fetch image records formatted for VisualTrainer.train_on_admin_images().
        Optionally filter by class_label (to train a single new class).
        """
        filters = [AdminTrainingImage.site_id == site_id]
        if class_label:
            filters.append(AdminTrainingImage.class_label == class_label)

        result = await db.execute(
            select(AdminTrainingImage).where(and_(*filters))
        )
        images = result.scalars().all()
        return [
            {"s3_key": img.s3_key, "class_label": img.class_label}
            for img in images
        ]

    async def mark_images_used(
        self,
        db: AsyncSession,
        site_id: str,
        job_id: str,
        class_label: str | None = None,
    ) -> int:
        """Mark images as used in a training job. Called after job completes."""
        filters = [
            AdminTrainingImage.site_id == site_id,
            AdminTrainingImage.is_used_in_training == False,
        ]
        if class_label:
            filters.append(AdminTrainingImage.class_label == class_label)

        result = await db.execute(
            select(AdminTrainingImage).where(and_(*filters))
        )
        images = result.scalars().all()
        count = 0
        for img in images:
            img.is_used_in_training = True
            img.used_in_training_job_id = job_id
            count += 1
        await db.commit()
        return count

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_image(
        self,
        db: AsyncSession,
        image_id: str,
        site_id: str,
        acting_user: User,
    ) -> None:
        """Delete a single training image (S3 + DB)."""
        from app.core.auth_service import auth_service
        auth_service.require_role(acting_user, UserRole.ADMIN)
        auth_service.require_site_access(acting_user, site_id)

        result = await db.execute(
            select(AdminTrainingImage).where(
                AdminTrainingImage.id == image_id,
                AdminTrainingImage.site_id == site_id,
            )
        )
        img = result.scalar_one_or_none()
        if not img:
            raise ValueError(f"Image {image_id} not found")
        if img.is_used_in_training:
            raise ValueError("Cannot delete image already used in training")

        # Delete from S3
        try:
            storage.delete(settings.S3_BUCKET_ADMIN_IMAGES, img.s3_key)
        except Exception as e:
            log.warning(f"S3 delete failed for {img.s3_key}: {e}")

        await db.delete(img)
        await db.commit()
        log.info(f"Admin image deleted: {image_id} (class={img.class_label})")

    async def delete_class(
        self,
        db: AsyncSession,
        site_id: str,
        class_label: str,
        acting_user: User,
    ) -> int:
        """Delete all images for a class. Fails if any were used in training."""
        from app.core.auth_service import auth_service
        auth_service.require_role(acting_user, UserRole.ADMIN)
        auth_service.require_site_access(acting_user, site_id)

        result = await db.execute(
            select(AdminTrainingImage).where(
                AdminTrainingImage.site_id == site_id,
                AdminTrainingImage.class_label == class_label,
            )
        )
        images = result.scalars().all()
        used = [img for img in images if img.is_used_in_training]
        if used:
            raise ValueError(
                f"{len(used)} images already used in training. "
                "Cannot delete class with trained images."
            )

        count = 0
        for img in images:
            try:
                storage.delete(settings.S3_BUCKET_ADMIN_IMAGES, img.s3_key)
            except Exception as e:
                log.warning(f"S3 delete failed: {e}")
            await db.delete(img)
            count += 1

        await db.commit()
        log.info(f"Deleted class '{class_label}': {count} images removed")
        return count

    # ── Presigned URLs for dashboard preview ──────────────────────────────────

    async def get_image_preview_url(
        self,
        db: AsyncSession,
        image_id: str,
        site_id: str,
    ) -> str:
        result = await db.execute(
            select(AdminTrainingImage).where(
                AdminTrainingImage.id == image_id,
                AdminTrainingImage.site_id == site_id,
            )
        )
        img = result.scalar_one_or_none()
        if not img:
            raise ValueError(f"Image {image_id} not found")
        return storage._presign(settings.S3_BUCKET_ADMIN_IMAGES, img.s3_key, 3600)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _count_class_images(
        self,
        db: AsyncSession,
        site_id: str,
        class_label: str,
    ) -> int:
        result = await db.execute(
            select(func.count()).select_from(AdminTrainingImage).where(
                AdminTrainingImage.site_id == site_id,
                AdminTrainingImage.class_label == class_label,
            )
        )
        return result.scalar() or 0

    @staticmethod
    def _validate_image(data: bytes) -> None:
        """Check magic bytes for JPEG or PNG."""
        if data[:2] == b"\xff\xd8":
            return  # JPEG
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return  # PNG
        raise ValueError("File must be a JPEG or PNG image")


# ── Singleton ─────────────────────────────────────────────────────────────────
admin_image_service = AdminImageService()
