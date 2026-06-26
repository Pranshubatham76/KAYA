"""
SentinelSite — SQLAlchemy ORM Models
All tables. No magic, no inheritance hell. Each table documented inline.
"""
import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, Enum, JSON, BigInteger, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


# ── Base ─────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────────────────

class UserRole(str, PyEnum):
    WORKER = "worker"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"
    SYSTEM = "system"


class EventStatus(str, PyEnum):
    PENDING = "pending"        # uploaded, not yet reviewed
    CONFIRMED = "confirmed"    # supervisor confirmed near-miss
    DISMISSED = "dismissed"    # supervisor dismissed as false positive


class OshaCategory(str, PyEnum):
    FALL = "fall"
    STRUCK_BY = "struck_by"
    CAUGHT_IN = "caught_in"
    ELECTROCUTION = "electrocution"
    OTHER = "other"


class SeverityLevel(str, PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelType(str, PyEnum):
    ACOUSTIC = "acoustic"
    VISUAL = "visual"


class ModelStatus(str, PyEnum):
    TRAINING = "training"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class TrainingJobStatus(str, PyEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, PyEnum):
    STRUCTURAL = "structural"
    SAFETY = "safety"
    SCHEDULE = "schedule"
    MATERIAL = "material"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    INSPECTION = "inspection"
    GENERAL = "general"


class DocumentStatus(str, PyEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ── Tables ───────────────────────────────────────────────────────────────────

class Site(Base):
    """
    Construction site. All data is namespaced by site_id.
    This is the top-level isolation unit.
    """
    __tablename__ = "sites"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    location = Column(String(500))
    timezone = Column(String(64), default="UTC")
    is_active = Column(Boolean, default=True)

    # Per-site calibrated thresholds (overrides global defaults)
    acoustic_threshold = Column(Float, nullable=True)   # θ₁
    imu_threshold = Column(Float, nullable=True)         # θ₂

    # GPS bounding box for the site (used for zone mapping)
    gps_lat_min = Column(Float, nullable=True)
    gps_lat_max = Column(Float, nullable=True)
    gps_lon_min = Column(Float, nullable=True)
    gps_lon_max = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    users = relationship("User", back_populates="site")
    events = relationship("NearMissEvent", back_populates="site")
    documents = relationship("SiteDocument", back_populates="site")
    model_versions = relationship("ModelVersion", back_populates="site")
    training_samples = relationship("TrainingSample", back_populates="site")
    zones = relationship("SiteZone", back_populates="site")


class SiteZone(Base):
    """
    Named geographic zone within a site (e.g. "Zone B - North Scaffolding").
    GPS coords → zone label via polygon lookup.
    """
    __tablename__ = "site_zones"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    name = Column(String(255), nullable=False)
    # GeoJSON polygon stored as JSONB for flexibility
    polygon_geojson = Column(JSONB, nullable=True)
    risk_weight = Column(Float, default=1.0)  # multiplier for risk scoring
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    site = relationship("Site", back_populates="zones")

    __table_args__ = (
        Index("ix_site_zones_site_id", "site_id"),
    )


class User(Base):
    """
    All roles in one table. role column gates what they can see/do.
    Workers are identified by worker_id (anonymizable SHA-256 hash).
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    email = Column(String(255), unique=True, nullable=True)    # null for workers
    worker_id = Column(String(64), nullable=True)               # SHA-256 hash
    hashed_password = Column(String(255), nullable=True)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.WORKER)
    is_active = Column(Boolean, default=True)
    device_id = Column(String(128), nullable=True)              # paired Android device
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    site = relationship("Site", back_populates="users")
    events = relationship("NearMissEvent", back_populates="worker")

    __table_args__ = (
        Index("ix_users_site_id", "site_id"),
        Index("ix_users_worker_id", "worker_id"),
    )


class NearMissEvent(Base):
    """
    Core table. One row per detected near-miss event.
    Lifecycle: PENDING → CONFIRMED | DISMISSED
    CONFIRMED events flow into training_samples.
    """
    __tablename__ = "near_miss_events"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    worker_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    device_id = Column(String(128), nullable=True)

    # Temporal
    event_timestamp = Column(DateTime(timezone=True), nullable=False)  # when it happened
    received_at = Column(DateTime(timezone=True), server_default=func.now())  # when we got it

    # Location
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    zone_id = Column(UUID(as_uuid=False), ForeignKey("site_zones.id"), nullable=True)

    # Acoustic signal
    yamnet_class = Column(String(128), nullable=True)        # e.g. "Impact, heavy objects"
    yamnet_class_id = Column(Integer, nullable=True)         # AudioSet index
    yamnet_confidence = Column(Float, nullable=True)         # 0.0–1.0
    anomaly_score = Column(Float, nullable=True)             # computed by AnomalyScorer

    # IMU signal
    imu_jerk_magnitude = Column(Float, nullable=True)        # rad/s²
    imu_timestamp_delta_ms = Column(Integer, nullable=True)  # Δt from audio event

    # Visual
    visual_class = Column(String(128), nullable=True)
    visual_confidence = Column(Float, nullable=True)
    frame_description = Column(Text, nullable=True)          # GPT-4o Vision output

    # Storage refs (S3 paths, not presigned URLs — generate those on demand)
    audio_s3_key = Column(String(512), nullable=True)        # 30s clip
    frame_s3_key = Column(String(512), nullable=True)        # single JPEG

    # Review
    status = Column(Enum(EventStatus), default=EventStatus.PENDING, nullable=False)
    reviewed_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    osha_category = Column(Enum(OshaCategory), nullable=True)
    severity = Column(Enum(SeverityLevel), nullable=True)
    review_notes = Column(Text, nullable=True)

    # Server-side recheck (YAMNet full quality)
    server_yamnet_class = Column(String(128), nullable=True)
    server_yamnet_confidence = Column(Float, nullable=True)
    server_recheck_done = Column(Boolean, default=False)

    # Relationships
    site = relationship("Site", back_populates="events")
    worker = relationship("User", foreign_keys=[worker_id], back_populates="events")
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    training_sample = relationship("TrainingSample", back_populates="event", uselist=False)
    zone = relationship("SiteZone")

    __table_args__ = (
        Index("ix_events_site_id", "site_id"),
        Index("ix_events_status", "status"),
        Index("ix_events_timestamp", "event_timestamp"),
        Index("ix_events_site_status", "site_id", "status"),
    )


class TrainingSample(Base):
    """
    Confirmed near-miss events eligible for model training.
    Created automatically when supervisor clicks Confirm.
    Used by ExperienceReplayBuffer.
    """
    __tablename__ = "training_samples"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    event_id = Column(
        UUID(as_uuid=False),
        ForeignKey("near_miss_events.id"),
        unique=True,  # one sample per event
        nullable=False,
    )

    # Labels (from supervisor review)
    label_acoustic = Column(String(128), nullable=True)   # yamnet class name
    label_visual = Column(String(128), nullable=True)     # visual class name
    label_osha = Column(Enum(OshaCategory), nullable=True)
    label_severity = Column(Enum(SeverityLevel), nullable=True)

    # Training lifecycle
    is_used_in_training = Column(Boolean, default=False)
    used_in_training_job_id = Column(UUID(as_uuid=False), ForeignKey("training_jobs.id"), nullable=True)
    training_used_at = Column(DateTime(timezone=True), nullable=True)

    # For uncertainty-weighted replay sampling
    # Updated after each model evaluation pass
    last_model_confidence = Column(Float, nullable=True)
    last_evaluated_at = Column(DateTime(timezone=True), nullable=True)

    # S3 refs (same as parent event — stored here for fast access without JOIN)
    audio_s3_key = Column(String(512), nullable=True)
    frame_s3_key = Column(String(512), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    site = relationship("Site", back_populates="training_samples")
    event = relationship("NearMissEvent", back_populates="training_sample")
    training_job = relationship("TrainingJob", back_populates="samples")

    __table_args__ = (
        Index("ix_training_samples_site_id", "site_id"),
        Index("ix_training_samples_used", "is_used_in_training"),
        Index("ix_training_samples_label_acoustic", "label_acoustic"),
    )


class ModelVersion(Base):
    """
    Every model artifact ever produced. Supports rollback.
    Only one ACTIVE model per (site, model_type) at a time.
    """
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    model_type = Column(Enum(ModelType), nullable=False)
    status = Column(Enum(ModelStatus), nullable=False, default=ModelStatus.TRAINING)

    # Versioning
    version = Column(Integer, nullable=False)  # monotonically increasing per site+type
    version_tag = Column(String(64), nullable=True)  # human label, e.g. "v3-week6"

    # Artifact location
    s3_key = Column(String(512), nullable=True)        # .tflite file
    s3_key_pytorch = Column(String(512), nullable=True)  # .pt checkpoint (for rollback)
    file_size_bytes = Column(BigInteger, nullable=True)

    # Accuracy metrics
    val_accuracy = Column(Float, nullable=True)
    previous_val_accuracy = Column(Float, nullable=True)
    per_class_accuracy = Column(JSONB, nullable=True)   # {class_name: accuracy}
    n_training_samples = Column(Integer, nullable=True)
    n_val_samples = Column(Integer, nullable=True)

    # Training metadata
    training_job_id = Column(UUID(as_uuid=False), ForeignKey("training_jobs.id"), nullable=True)
    trained_at = Column(DateTime(timezone=True), nullable=True)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    retired_at = Column(DateTime(timezone=True), nullable=True)

    # Rollback tracking
    rolled_back_from_id = Column(UUID(as_uuid=False), ForeignKey("model_versions.id"), nullable=True)
    rollback_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    site = relationship("Site", back_populates="model_versions")
    training_job = relationship("TrainingJob", back_populates="model_version")

    __table_args__ = (
        Index("ix_model_versions_site_type_status", "site_id", "model_type", "status"),
        UniqueConstraint("site_id", "model_type", "version", name="uq_model_version"),
    )


class TrainingJob(Base):
    """
    Celery training job record. Tracks every triggered training run.
    """
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    model_type = Column(Enum(ModelType), nullable=False)
    trigger = Column(String(64), default="scheduler")  # "scheduler" | "admin" | "manual"
    status = Column(Enum(TrainingJobStatus), default=TrainingJobStatus.QUEUED)

    celery_task_id = Column(String(255), nullable=True)

    n_new_samples = Column(Integer, nullable=True)
    n_historical_samples = Column(Integer, nullable=True)
    replay_strategy = Column(String(64), default="class_balanced")

    # Timing
    queued_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    error_message = Column(Text, nullable=True)
    training_log = Column(JSONB, nullable=True)  # epoch-by-epoch metrics

    # Relationships (back_populates defined on other side)
    samples = relationship("TrainingSample", back_populates="training_job")
    model_version = relationship("ModelVersion", back_populates="training_job", uselist=False)

    __table_args__ = (
        Index("ix_training_jobs_site_id", "site_id"),
        Index("ix_training_jobs_status", "status"),
    )


class SiteDocument(Base):
    """
    Site document uploaded by admin for RAG ingestion.
    After ingestion, chunks live in Qdrant. This table tracks metadata.
    """
    __tablename__ = "site_documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    uploaded_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)

    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    doc_type = Column(Enum(DocumentType), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.QUEUED)

    # Storage
    s3_key = Column(String(512), nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)

    # Ingestion results
    n_chunks = Column(Integer, nullable=True)
    n_tokens_total = Column(Integer, nullable=True)
    qdrant_collection = Column(String(255), nullable=True)  # sentinel_{site_id}
    ingestion_error = Column(Text, nullable=True)

    # Versioning — replacing a document creates a new row, marks old as superseded
    supersedes_id = Column(UUID(as_uuid=False), ForeignKey("site_documents.id"), nullable=True)
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    indexed_at = Column(DateTime(timezone=True), nullable=True)

    site = relationship("Site", back_populates="documents")
    uploader = relationship("User")

    __table_args__ = (
        Index("ix_site_documents_site_id", "site_id"),
        Index("ix_site_documents_site_type", "site_id", "doc_type"),
        Index("ix_site_documents_status", "status"),
    )


class AdminTrainingImage(Base):
    """
    Images uploaded by admin for visual model fine-tuning.
    These are site-specific object classes (e.g. "confined_space_entrance").
    """
    __tablename__ = "admin_training_images"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    uploaded_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)

    s3_key = Column(String(512), nullable=False)
    filename = Column(String(500), nullable=False)
    class_label = Column(String(255), nullable=False)   # e.g. "unguarded_excavation"
    file_size_bytes = Column(Integer, nullable=True)

    # Training lifecycle
    is_used_in_training = Column(Boolean, default=False)
    used_in_training_job_id = Column(UUID(as_uuid=False), ForeignKey("training_jobs.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_admin_images_site_class", "site_id", "class_label"),
    )


class VoiceQueryLog(Base):
    """
    Optional log of voice queries (only if worker opts in — NFR-S01).
    Default is session-only; this table only gets rows for opted-in workers.
    """
    __tablename__ = "voice_query_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id = Column(UUID(as_uuid=False), ForeignKey("sites.id"), nullable=False)
    worker_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)

    query_text = Column(Text, nullable=False)
    detected_intent = Column(Enum(DocumentType), nullable=True)
    answer_text = Column(Text, nullable=True)
    source_document_ids = Column(ARRAY(String), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    was_answered = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_voice_logs_site_id", "site_id"),
    )
