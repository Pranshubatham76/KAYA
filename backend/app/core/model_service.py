"""
SentinelSite — Model Service
OTA model delivery, version management, post-deploy accuracy reporting,
auto-rollback orchestration.
Device calls: GET /models/latest → download_url → cold-start swap.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy import and_, desc, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.storage import storage
from app.db.models import (
    ModelStatus, ModelType, ModelVersion,
    TrainingJob, User, UserRole,
)
from app.training.model_pusher import model_pusher, ROLLBACK_THRESHOLD

log = logging.getLogger(__name__)


class ModelService:

    # ── OTA — device-facing ───────────────────────────────────────────────────

    async def check_for_update(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: str,
        device_current_version: int | None,
    ) -> dict:
        """
        Called by device on cold start.
        Returns {has_update, version, download_url, file_size_bytes}.
        Model swap only on cold start — never hot-swap mid-inference.
        """
        active = await self._get_active(db, site_id, ModelType(model_type))

        if not active or not active.s3_key:
            return {
                "has_update": False,
                "version": None,
                "download_url": None,
                "file_size_bytes": None,
                "val_accuracy": None,
            }

        has_update = (
            device_current_version is None
            or active.version > device_current_version
        )

        return {
            "has_update": has_update,
            "version": active.version,
            "download_url": (
                storage.presigned_model_url(active.s3_key)
                if has_update else None
            ),
            "file_size_bytes": active.file_size_bytes,
            "val_accuracy": active.val_accuracy,
        }

    async def report_post_deploy_accuracy(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: str,
        version: int,
        reported_accuracy: float,
    ) -> dict:
        """
        Device reports accuracy after deploying new model.
        Triggers auto-rollback if accuracy dropped >5%.
        """
        active = await self._get_active(db, site_id, ModelType(model_type))
        if not active:
            return {"action": "no_active_model"}

        if active.version != version:
            log.warning(
                f"Accuracy report for v{version} but active is v{active.version} "
                f"— ignoring (device may be stale)"
            )
            return {"action": "version_mismatch", "active_version": active.version}

        prev_acc = active.previous_val_accuracy or 0.0
        drop = prev_acc - reported_accuracy

        if drop > ROLLBACK_THRESHOLD:
            log.warning(
                f"Auto-rollback: site={site_id}, type={model_type}, "
                f"v{version}, drop={drop:.3f}"
            )
            # Execute rollback synchronously (must be fast <10min per NFR)
            result = self._execute_rollback_sync(
                site_id, ModelType(model_type), active, reported_accuracy
            )
            # Persist changes
            await db.commit()
            return result

        log.info(
            f"Post-deploy accuracy OK: site={site_id}, type={model_type}, "
            f"v{version}, acc={reported_accuracy:.3f}"
        )
        return {
            "action": "ok",
            "version": version,
            "reported_accuracy": reported_accuracy,
        }

    def _execute_rollback_sync(
        self,
        site_id: str,
        model_type: ModelType,
        current: ModelVersion,
        trigger_accuracy: float,
    ) -> dict:
        """Synchronous rollback — uses sync DB session for speed."""
        from app.db.session import get_sync_db
        with get_sync_db() as sync_db:
            # Find previous retired model
            from sqlalchemy import select as sync_select
            prev = (
                sync_db.query(ModelVersion)
                .filter(
                    ModelVersion.site_id == site_id,
                    ModelVersion.model_type == model_type,
                    ModelVersion.status == ModelStatus.RETIRED,
                    ModelVersion.version < current.version,
                )
                .order_by(desc(ModelVersion.version))
                .first()
            )

            current_row = sync_db.query(ModelVersion).get(current.id)
            current_row.status = ModelStatus.ROLLED_BACK
            current_row.rollback_reason = (
                f"Post-deploy accuracy {trigger_accuracy:.3f} dropped "
                f"{(current.previous_val_accuracy or 0) - trigger_accuracy:.3f} "
                f"> threshold {ROLLBACK_THRESHOLD}"
            )

            if prev:
                prev.status = ModelStatus.ACTIVE
                prev.deployed_at = datetime.utcnow()
                sync_db.commit()
                download_url = storage.presigned_model_url(prev.s3_key)
                log.info(f"Rollback complete: v{current.version} → v{prev.version}")
                return {
                    "action": "rollback",
                    "rolled_back_from": current.version,
                    "rolled_back_to": prev.version,
                    "download_url": download_url,
                }
            else:
                current_row.status = ModelStatus.FAILED
                sync_db.commit()
                log.error(f"Rollback failed — no previous model for {site_id}/{model_type.value}")
                return {
                    "action": "rollback_failed",
                    "reason": "No previous model version available",
                }

    # ── Version history ───────────────────────────────────────────────────────

    async def list_versions(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: str | None = None,
        limit: int = 20,
    ) -> list[ModelVersion]:
        filters = [ModelVersion.site_id == site_id]
        if model_type:
            filters.append(ModelVersion.model_type == ModelType(model_type))

        result = await db.execute(
            select(ModelVersion)
            .where(and_(*filters))
            .order_by(desc(ModelVersion.version))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_active_versions(
        self,
        db: AsyncSession,
        site_id: str,
    ) -> dict[str, ModelVersion | None]:
        """Returns {acoustic: ModelVersion|None, visual: ModelVersion|None}."""
        return {
            "acoustic": await self._get_active(db, site_id, ModelType.ACOUSTIC),
            "visual": await self._get_active(db, site_id, ModelType.VISUAL),
        }

    async def get_version(
        self,
        db: AsyncSession,
        site_id: str,
        version_id: str,
    ) -> ModelVersion | None:
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.id == version_id,
                ModelVersion.site_id == site_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_download_url(
        self,
        db: AsyncSession,
        site_id: str,
        version_id: str,
    ) -> str | None:
        """Generate presigned download URL for a specific model version."""
        version = await self.get_version(db, site_id, version_id)
        if not version or not version.s3_key:
            return None
        return storage.presigned_model_url(version.s3_key)

    async def get_accuracy_history(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: str,
        last_n: int = 10,
    ) -> list[dict]:
        """
        Accuracy trend over last N model versions.
        Used by dashboard analytics chart.
        """
        result = await db.execute(
            select(ModelVersion)
            .where(
                ModelVersion.site_id == site_id,
                ModelVersion.model_type == ModelType(model_type),
                ModelVersion.val_accuracy.isnot(None),
            )
            .order_by(desc(ModelVersion.version))
            .limit(last_n)
        )
        versions = list(result.scalars().all())
        versions.reverse()  # chronological order

        return [
            {
                "version": v.version,
                "val_accuracy": v.val_accuracy,
                "previous_val_accuracy": v.previous_val_accuracy,
                "n_training_samples": v.n_training_samples,
                "trained_at": v.trained_at.isoformat() if v.trained_at else None,
                "status": v.status.value,
                "per_class_accuracy": v.per_class_accuracy,
            }
            for v in versions
        ]

    async def admin_manual_rollback(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: str,
        target_version_id: str,
        acting_user: User,
    ) -> dict:
        """
        Admin manually rolls back to a specific model version.
        Used when automatic rollback isn't enough or wrong version promoted.
        """
        from app.core.auth_service import auth_service
        auth_service.require_role(acting_user, UserRole.ADMIN)
        auth_service.require_site_access(acting_user, site_id)

        target = await self.get_version(db, site_id, target_version_id)
        if not target:
            raise ValueError(f"Model version {target_version_id} not found")
        if target.model_type != ModelType(model_type):
            raise ValueError("Model type mismatch")
        if not target.s3_key:
            raise ValueError("Target version has no artifact (cannot deploy)")

        # Retire current active
        current = await self._get_active(db, site_id, ModelType(model_type))
        if current:
            current.status = ModelStatus.RETIRED
            current.retired_at = datetime.utcnow()

        # Promote target
        target.status = ModelStatus.ACTIVE
        target.deployed_at = datetime.utcnow()
        target.rolled_back_from_id = current.id if current else None
        target.rollback_reason = f"Manual rollback by admin {acting_user.id}"
        await db.commit()

        download_url = storage.presigned_model_url(target.s3_key)
        log.info(
            f"Manual rollback: site={site_id}, type={model_type}, "
            f"→ v{target.version} by admin={acting_user.id}"
        )
        return {
            "action": "manual_rollback",
            "rolled_back_to_version": target.version,
            "download_url": download_url,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _get_active(
        self,
        db: AsyncSession,
        site_id: str,
        model_type: ModelType,
    ) -> ModelVersion | None:
        result = await db.execute(
            select(ModelVersion).where(
                ModelVersion.site_id == site_id,
                ModelVersion.model_type == model_type,
                ModelVersion.status == ModelStatus.ACTIVE,
            )
        )
        return result.scalar_one_or_none()


# ── Singleton ─────────────────────────────────────────────────────────────────
model_service = ModelService()
