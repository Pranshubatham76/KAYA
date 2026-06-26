"""
SentinelSite — Model Pusher & Rollback Service
Handles OTA model delivery to devices and auto-rollback on accuracy drop.
Rollback trigger: post-deploy accuracy drops >5% → revert in <10min.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from app.config import settings
from app.core.storage import storage
from app.db.models import ModelStatus, ModelType, ModelVersion

log = logging.getLogger(__name__)

ROLLBACK_THRESHOLD = 0.05  # 5% accuracy drop triggers rollback


class ModelPusher:
    """
    Manages model promotion, OTA delivery, and rollback.
    Called after training job completes successfully.
    """

    def get_latest_active(
        self,
        db,
        site_id: str,
        model_type: ModelType,
    ) -> ModelVersion | None:
        """Get the currently deployed model for a site+type."""
        return (
            db.query(ModelVersion)
            .filter(
                ModelVersion.site_id == site_id,
                ModelVersion.model_type == model_type,
                ModelVersion.status == ModelStatus.ACTIVE,
            )
            .first()
        )

    def get_download_url(
        self,
        db,
        site_id: str,
        model_type: str,
        device_current_version: int | None = None,
    ) -> dict:
        """
        Called by device on startup to check for model updates.
        Returns {has_update, version, download_url, file_size_bytes}.
        """
        from app.db.models import ModelType as MT

        active = self.get_latest_active(db, site_id, MT(model_type))
        if not active or not active.s3_key:
            return {"has_update": False, "version": None, "download_url": None}

        has_update = (
            device_current_version is None
            or active.version > device_current_version
        )
        download_url = (
            storage.presigned_model_url(active.s3_key)
            if has_update else None
        )
        return {
            "has_update": has_update,
            "version": active.version,
            "download_url": download_url,
            "file_size_bytes": active.file_size_bytes,
            "val_accuracy": active.val_accuracy,
        }

    def check_and_rollback(
        self,
        db,
        site_id: str,
        model_type: ModelType,
        reported_accuracy: float,
    ) -> dict:
        """
        Device reports post-deploy accuracy.
        If it dropped >5% vs previous model, auto-rollback.
        Returns {action: "rollback"|"ok", ...}
        """
        active = self.get_latest_active(db, site_id, model_type)
        if not active:
            return {"action": "no_model"}

        prev_acc = active.previous_val_accuracy or 0.0
        drop = prev_acc - reported_accuracy

        if drop > ROLLBACK_THRESHOLD:
            log.warning(
                f"Auto-rollback triggered: site={site_id}, type={model_type.value}, "
                f"prev_acc={prev_acc:.3f}, reported={reported_accuracy:.3f}, drop={drop:.3f}"
            )
            return self._execute_rollback(db, site_id, model_type, active, reported_accuracy)

        log.info(
            f"Post-deploy accuracy OK: {reported_accuracy:.3f} "
            f"(prev={prev_acc:.3f}, drop={drop:.3f})"
        )
        return {"action": "ok", "accuracy": reported_accuracy}

    def _execute_rollback(
        self,
        db,
        site_id: str,
        model_type: ModelType,
        current: ModelVersion,
        trigger_accuracy: float,
    ) -> dict:
        """
        Revert to previous active model.
        Marks current as ROLLED_BACK, promotes previous RETIRED model.
        """
        # Find the most recent retired model (the one we replaced)
        prev_version = (
            db.query(ModelVersion)
            .filter(
                ModelVersion.site_id == site_id,
                ModelVersion.model_type == model_type,
                ModelVersion.status == ModelStatus.RETIRED,
                ModelVersion.version < current.version,
            )
            .order_by(ModelVersion.version.desc())
            .first()
        )

        # Mark current as rolled back
        current.status = ModelStatus.ROLLED_BACK
        current.rollback_reason = (
            f"Post-deploy accuracy {trigger_accuracy:.3f} dropped "
            f"{(current.previous_val_accuracy or 0) - trigger_accuracy:.3f} "
            f"below threshold {ROLLBACK_THRESHOLD}"
        )

        if prev_version:
            # Re-promote previous version
            prev_version.status = ModelStatus.ACTIVE
            prev_version.deployed_at = datetime.utcnow()
            db.commit()

            download_url = storage.presigned_model_url(prev_version.s3_key)
            log.info(f"Rollback complete → v{prev_version.version}")
            return {
                "action": "rollback",
                "rolled_back_from": current.version,
                "rolled_back_to": prev_version.version,
                "download_url": download_url,
            }
        else:
            # No previous model — mark as FAILED, notify supervisors
            current.status = ModelStatus.FAILED
            db.commit()
            log.error(f"Rollback failed — no previous model available for {site_id}/{model_type.value}")
            return {
                "action": "rollback_failed",
                "reason": "No previous model version available",
            }


# ── Singleton ─────────────────────────────────────────────────────────────────
model_pusher = ModelPusher()
