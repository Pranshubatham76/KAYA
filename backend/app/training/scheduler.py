"""
SentinelSite — Training Scheduler & Celery Tasks
APScheduler checks every 6h for training trigger conditions.
Celery executes the actual training jobs (GPU workers).

Trigger conditions (ALL must be true — FR-L02):
  a) ≥ 20 new confirmed samples since last training
  b) ≥ 7 days since last training run
  c) GPU utilization < 50%
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from celery import Celery
from celery.utils.log import get_task_logger

from app.config import settings

log = logging.getLogger(__name__)
task_log = get_task_logger(__name__)

# ── Celery app ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "sentinelsite",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,           # re-queue on worker crash
    worker_prefetch_multiplier=1,  # one task at a time on GPU worker
    task_routes={
        "app.training.scheduler.train_acoustic_task": {"queue": "gpu"},
        "app.training.scheduler.train_visual_task": {"queue": "gpu"},
        "app.training.scheduler.ingest_document_task": {"queue": "cpu"},
        "app.training.scheduler.recheck_event_task": {"queue": "cpu"},
        "app.training.scheduler.push_model_task": {"queue": "cpu"},
    },
)


# ── Trigger condition checks ──────────────────────────────────────────────────

def _count_new_samples_since_last_training(db, site_id: str, model_type: str) -> int:
    """Count confirmed samples not yet used in training for this site+model_type."""
    from app.db.models import ModelType, TrainingSample

    mt = ModelType(model_type)
    label_field = (
        TrainingSample.label_acoustic if mt == ModelType.ACOUSTIC
        else TrainingSample.label_visual
    )
    count = (
        db.query(TrainingSample)
        .filter(
            TrainingSample.site_id == site_id,
            TrainingSample.is_used_in_training == False,
            label_field.isnot(None),
        )
        .count()
    )
    return count


def _days_since_last_training(db, site_id: str, model_type: str) -> float:
    """Days elapsed since last completed training job for this site+type."""
    from app.db.models import ModelType, TrainingJob, TrainingJobStatus

    last_job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.site_id == site_id,
            TrainingJob.model_type == ModelType(model_type),
            TrainingJob.status == TrainingJobStatus.COMPLETED,
        )
        .order_by(TrainingJob.completed_at.desc())
        .first()
    )
    if last_job is None or last_job.completed_at is None:
        return float("inf")  # never trained → trigger immediately
    delta = datetime.utcnow() - last_job.completed_at.replace(tzinfo=None)
    return delta.total_seconds() / 86400


def _gpu_utilization() -> float:
    """Get current GPU utilization (0.0–1.0). Returns 0.0 if no GPU."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            util = float(result.stdout.strip().split("\n")[0]) / 100.0
            return util
    except Exception:
        pass
    return 0.0


def check_training_conditions(db, site_id: str, model_type: str) -> tuple[bool, dict]:
    """
    Check all 3 trigger conditions.
    Returns (should_trigger, debug_info).
    """
    n_new = _count_new_samples_since_last_training(db, site_id, model_type)
    days_since = _days_since_last_training(db, site_id, model_type)
    gpu_util = _gpu_utilization()

    cond_a = n_new >= settings.TRAINING_MIN_NEW_SAMPLES
    cond_b = days_since >= settings.TRAINING_MIN_INTERVAL_DAYS
    cond_c = gpu_util < settings.TRAINING_MAX_GPU_UTILIZATION

    debug = {
        "n_new_samples": n_new,
        "min_required": settings.TRAINING_MIN_NEW_SAMPLES,
        "days_since_last_training": round(days_since, 2),
        "min_interval_days": settings.TRAINING_MIN_INTERVAL_DAYS,
        "gpu_utilization": round(gpu_util, 3),
        "max_gpu_util": settings.TRAINING_MAX_GPU_UTILIZATION,
        "cond_a_samples": cond_a,
        "cond_b_interval": cond_b,
        "cond_c_gpu": cond_c,
        "should_trigger": cond_a and cond_b and cond_c,
    }
    log.info(f"Training conditions [{site_id}/{model_type}]: {debug}")
    return cond_a and cond_b and cond_c, debug


# ── APScheduler beat task ─────────────────────────────────────────────────────

@celery_app.task(name="app.training.scheduler.check_training_schedule")
def check_training_schedule() -> dict:
    """
    Runs every 6 hours via Celery Beat.
    Checks all active sites × both model types.
    Enqueues training tasks if conditions met.
    """
    from app.db.session import get_sync_db
    from app.db.models import ModelType, Site

    triggered = []

    with get_sync_db() as db:
        sites = db.query(Site).filter(Site.is_active == True).all()
        for site in sites:
            for model_type in [ModelType.ACOUSTIC, ModelType.VISUAL]:
                should_trigger, debug = check_training_conditions(
                    db, str(site.id), model_type.value
                )
                if should_trigger:
                    job_id = _enqueue_training_job(
                        db=db,
                        site_id=str(site.id),
                        model_type=model_type.value,
                        trigger="scheduler",
                    )
                    triggered.append({
                        "site_id": str(site.id),
                        "model_type": model_type.value,
                        "job_id": job_id,
                        **debug,
                    })
                    log.info(f"Training enqueued: site={site.id}, type={model_type.value}, job={job_id}")

    log.info(f"Scheduler run complete: {len(triggered)} jobs triggered")
    return {"triggered_jobs": triggered, "checked_at": datetime.utcnow().isoformat()}


def _enqueue_training_job(
    db,
    site_id: str,
    model_type: str,
    trigger: str = "scheduler",
    replay_strategy: str = "class_balanced",
) -> str:
    """Create TrainingJob DB record and dispatch Celery task."""
    from app.db.models import ModelType, TrainingJob, TrainingJobStatus

    job = TrainingJob(
        id=str(uuid4()),
        site_id=site_id,
        model_type=ModelType(model_type),
        trigger=trigger,
        status=TrainingJobStatus.QUEUED,
        replay_strategy=replay_strategy,
    )
    db.add(job)
    db.flush()
    job_id = str(job.id)

    # Dispatch to GPU queue
    task_fn = train_acoustic_task if model_type == "acoustic" else train_visual_task
    celery_task = task_fn.apply_async(
        kwargs={"job_id": job_id, "site_id": site_id},
        queue="gpu",
    )
    job.celery_task_id = celery_task.id
    db.commit()
    return job_id


# ── Celery Training Tasks ─────────────────────────────────────────────────────

@celery_app.task(
    name="app.training.scheduler.train_acoustic_task",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def train_acoustic_task(self, job_id: str, site_id: str) -> dict[str, Any]:
    """
    Celery task: acoustic model training.
    Runs on GPU worker queue.
    """
    from app.db.models import (
        ModelStatus, ModelType, ModelVersion, TrainingJob,
        TrainingJobStatus, TrainingSample,
    )
    from app.db.session import get_sync_db
    from app.training.acoustic_trainer import YAMNET_TARGET_CLASSES, AcousticTrainer
    from app.training.replay_buffer import ExperienceReplayBuffer
    from app.core.storage import storage

    task_log.info(f"Starting acoustic training: job={job_id}, site={site_id}")

    with get_sync_db() as db:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            raise ValueError(f"TrainingJob {job_id} not found")

        job.status = TrainingJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()

        try:
            # 1. Get new (untrained) samples
            new_samples_db = (
                db.query(TrainingSample)
                .filter(
                    TrainingSample.site_id == site_id,
                    TrainingSample.is_used_in_training == False,
                    TrainingSample.label_acoustic.isnot(None),
                    TrainingSample.audio_s3_key.isnot(None),
                )
                .all()
            )
            new_sample_ids = [str(s.id) for s in new_samples_db]
            job.n_new_samples = len(new_sample_ids)

            # 2. Build replay batch (70% historical + 30% new)
            buffer = ExperienceReplayBuffer(db)
            replay_batch = buffer.build_batch(
                site_id=site_id,
                new_sample_ids=new_sample_ids,
                strategy=job.replay_strategy,
            )
            job.n_historical_samples = len(replay_batch) - len(new_sample_ids)
            db.commit()

            # 3. Train
            trainer = AcousticTrainer(site_id=site_id)
            results = trainer.train(replay_batch)

            # 4. Promotion gate — new model val_acc ≥ previous
            prev_model = (
                db.query(ModelVersion)
                .filter(
                    ModelVersion.site_id == site_id,
                    ModelVersion.model_type == ModelType.ACOUSTIC,
                    ModelVersion.status == ModelStatus.ACTIVE,
                )
                .first()
            )
            prev_acc = prev_model.val_accuracy if prev_model else 0.0
            if results["val_accuracy"] < prev_acc:
                task_log.warning(
                    f"Promotion gate FAILED: new={results['val_accuracy']:.3f} < "
                    f"prev={prev_acc:.3f}. Discarding model."
                )
                job.status = TrainingJobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.error_message = (
                    f"Promotion gate failed: {results['val_accuracy']:.3f} < {prev_acc:.3f}"
                )
                job.training_log = results["history"]
                db.commit()
                return {"promoted": False, "val_accuracy": results["val_accuracy"]}

            # 5. Get next version number
            max_version = (
                db.query(ModelVersion)
                .filter(
                    ModelVersion.site_id == site_id,
                    ModelVersion.model_type == ModelType.ACOUSTIC,
                )
                .count()
            ) + 1

            # 6. Upload to S3
            tflite_key, pt_key = storage.upload_model(
                site_id=site_id,
                model_type="acoustic",
                version=max_version,
                tflite_data=results["tflite_bytes"],
                pytorch_data=results.get("pytorch_bytes"),
            )

            # 7. Create ModelVersion record
            new_version = ModelVersion(
                id=str(uuid4()),
                site_id=site_id,
                model_type=ModelType.ACOUSTIC,
                status=ModelStatus.ACTIVE,
                version=max_version,
                s3_key=tflite_key,
                s3_key_pytorch=pt_key,
                val_accuracy=results["val_accuracy"],
                previous_val_accuracy=prev_acc,
                per_class_accuracy=results.get("per_class_accuracy"),
                n_training_samples=results["n_train"],
                n_val_samples=results["n_val"],
                training_job_id=job_id,
                trained_at=datetime.utcnow(),
                deployed_at=datetime.utcnow(),
            )
            db.add(new_version)

            # 8. Retire previous active model
            if prev_model:
                prev_model.status = ModelStatus.RETIRED
                prev_model.retired_at = datetime.utcnow()

            # 9. Mark samples as used
            db.query(TrainingSample).filter(
                TrainingSample.id.in_(new_sample_ids)
            ).update(
                {
                    "is_used_in_training": True,
                    "used_in_training_job_id": job_id,
                    "training_used_at": datetime.utcnow(),
                },
                synchronize_session=False,
            )

            # 10. Finalize job
            job.status = TrainingJobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.training_log = results["history"]
            db.commit()

            task_log.info(
                f"Acoustic training COMPLETE: v{max_version}, "
                f"val_acc={results['val_accuracy']:.3f}"
            )
            return {
                "promoted": True,
                "version": max_version,
                "val_accuracy": results["val_accuracy"],
                "tflite_key": tflite_key,
            }

        except Exception as exc:
            task_log.error(f"Acoustic training FAILED: {exc}", exc_info=True)
            job.status = TrainingJobStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()
            db.commit()
            raise self.retry(exc=exc) from exc


@celery_app.task(
    name="app.training.scheduler.train_visual_task",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def train_visual_task(self, job_id: str, site_id: str) -> dict[str, Any]:
    """Celery task: visual model training."""
    from app.db.models import (
        ModelStatus, ModelType, ModelVersion, TrainingJob,
        TrainingJobStatus, TrainingSample,
    )
    from app.db.session import get_sync_db
    from app.training.replay_buffer import ExperienceReplayBuffer
    from app.training.visual_trainer import VisualTrainer
    from app.core.storage import storage
    from collections import Counter

    task_log.info(f"Starting visual training: job={job_id}, site={site_id}")

    with get_sync_db() as db:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            raise ValueError(f"TrainingJob {job_id} not found")

        job.status = TrainingJobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()

        try:
            # Build class_to_idx from all known visual labels for this site
            all_labels = (
                db.query(TrainingSample.label_visual)
                .filter(
                    TrainingSample.site_id == site_id,
                    TrainingSample.label_visual.isnot(None),
                )
                .distinct()
                .all()
            )
            labels = sorted([r[0] for r in all_labels])
            class_to_idx = {cls: i for i, cls in enumerate(labels)}

            if not class_to_idx:
                raise ValueError("No visual labels found for site")

            # New (untrained) samples
            new_samples_db = (
                db.query(TrainingSample)
                .filter(
                    TrainingSample.site_id == site_id,
                    TrainingSample.is_used_in_training == False,
                    TrainingSample.label_visual.isnot(None),
                    TrainingSample.frame_s3_key.isnot(None),
                )
                .all()
            )
            new_sample_ids = [str(s.id) for s in new_samples_db]
            job.n_new_samples = len(new_sample_ids)

            # Replay batch
            buffer = ExperienceReplayBuffer(db)
            replay_batch = buffer.build_batch(
                site_id=site_id,
                new_sample_ids=new_sample_ids,
                strategy=job.replay_strategy,
            )
            job.n_historical_samples = len(replay_batch) - len(new_sample_ids)
            db.commit()

            # Train
            trainer = VisualTrainer(site_id=site_id, class_to_idx=class_to_idx)
            results = trainer.train_on_replay(replay_batch)

            # Promotion gate
            prev_model = (
                db.query(ModelVersion)
                .filter(
                    ModelVersion.site_id == site_id,
                    ModelVersion.model_type == ModelType.VISUAL,
                    ModelVersion.status == ModelStatus.ACTIVE,
                )
                .first()
            )
            prev_acc = prev_model.val_accuracy if prev_model else 0.0
            if results["val_accuracy"] < prev_acc:
                job.status = TrainingJobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.error_message = f"Promotion gate failed: {results['val_accuracy']:.3f} < {prev_acc:.3f}"
                job.training_log = results["history"]
                db.commit()
                return {"promoted": False, "val_accuracy": results["val_accuracy"]}

            max_version = (
                db.query(ModelVersion)
                .filter(ModelVersion.site_id == site_id, ModelVersion.model_type == ModelType.VISUAL)
                .count()
            ) + 1

            tflite_key, pt_key = storage.upload_model(
                site_id=site_id, model_type="visual", version=max_version,
                tflite_data=results["tflite_bytes"],
                pytorch_data=results.get("pytorch_bytes"),
            )

            new_version = ModelVersion(
                id=str(uuid4()),
                site_id=site_id,
                model_type=ModelType.VISUAL,
                status=ModelStatus.ACTIVE,
                version=max_version,
                s3_key=tflite_key,
                s3_key_pytorch=pt_key,
                val_accuracy=results["val_accuracy"],
                previous_val_accuracy=prev_acc,
                per_class_accuracy=results.get("per_class_accuracy"),
                n_training_samples=results["n_train"],
                n_val_samples=results["n_val"],
                training_job_id=job_id,
                trained_at=datetime.utcnow(),
                deployed_at=datetime.utcnow(),
            )
            db.add(new_version)

            if prev_model:
                prev_model.status = ModelStatus.RETIRED
                prev_model.retired_at = datetime.utcnow()

            db.query(TrainingSample).filter(
                TrainingSample.id.in_(new_sample_ids)
            ).update(
                {"is_used_in_training": True, "used_in_training_job_id": job_id,
                 "training_used_at": datetime.utcnow()},
                synchronize_session=False,
            )

            job.status = TrainingJobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.training_log = results["history"]
            db.commit()

            task_log.info(f"Visual training COMPLETE: v{max_version}, val_acc={results['val_accuracy']:.3f}")
            return {"promoted": True, "version": max_version, "val_accuracy": results["val_accuracy"]}

        except Exception as exc:
            task_log.error(f"Visual training FAILED: {exc}", exc_info=True)
            job.status = TrainingJobStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()
            db.commit()
            raise self.retry(exc=exc) from exc


# ── Document ingestion task ───────────────────────────────────────────────────

@celery_app.task(name="app.training.scheduler.ingest_document_task", bind=True, max_retries=2)
def ingest_document_task(self, doc_id: str, site_id: str) -> dict:
    """Celery task: ingest a site document into Qdrant."""
    import asyncio
    from app.db.models import DocumentStatus, SiteDocument
    from app.db.session import get_sync_db
    from app.rag.ingestion import ingest_document

    with get_sync_db() as db:
        doc = db.query(SiteDocument).filter(SiteDocument.id == doc_id).first()
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        doc.status = DocumentStatus.PROCESSING
        db.commit()

        try:
            result = asyncio.run(
                ingest_document(
                    site_id=site_id,
                    doc_id=doc_id,
                    doc_type=doc.doc_type,
                    filename=doc.original_filename,
                    s3_key=doc.s3_key,
                )
            )
            doc.status = DocumentStatus.INDEXED
            doc.n_chunks = result["n_chunks"]
            doc.n_tokens_total = result["n_tokens"]
            doc.qdrant_collection = f"sentinel_{str(site_id).replace('-', '')}"
            doc.indexed_at = datetime.utcnow()
            db.commit()
            task_log.info(f"Document ingested: {doc_id}, {result['n_chunks']} chunks")
            return result

        except Exception as exc:
            doc.status = DocumentStatus.FAILED
            doc.ingestion_error = str(exc)
            db.commit()
            raise self.retry(exc=exc) from exc


# ── Celery Beat schedule ──────────────────────────────────────────────────────

celery_app.conf.beat_schedule = {
    "check-training-schedule": {
        "task": "app.training.scheduler.check_training_schedule",
        "schedule": settings.TRAINING_CHECK_INTERVAL_HOURS * 3600,  # every 6h
        "options": {"queue": "cpu"},
    },
}
