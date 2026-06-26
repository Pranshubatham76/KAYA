"""
SentinelSite — Pydantic v2 Schemas
Request/response validation for every API surface.
Mirrors db/models.py but decoupled — DB models ≠ API shapes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.db.models import (
    DocumentType, EventStatus, ModelStatus, ModelType,
    OshaCategory, SeverityLevel, TrainingJobStatus, UserRole,
    DocumentStatus,
)


# ── Shared base ───────────────────────────────────────────────────────────────

class _Base(BaseModel):
    model_config = {"from_attributes": True}  # allow ORM → schema


# ── Site ─────────────────────────────────────────────────────────────────────

class SiteCreate(_Base):
    name: str = Field(..., min_length=2, max_length=255)
    location: str | None = None
    timezone: str = "UTC"
    acoustic_threshold: float | None = Field(None, ge=0.0, le=1.0)
    imu_threshold: float | None = Field(None, ge=0.0)


class SiteRead(_Base):
    id: UUID
    name: str
    location: str | None
    timezone: str
    is_active: bool
    acoustic_threshold: float | None
    imu_threshold: float | None
    created_at: datetime


class SiteThresholdUpdate(_Base):
    acoustic_threshold: float | None = Field(None, ge=0.0, le=1.0)
    imu_threshold: float | None = Field(None, ge=0.0)


# ── Auth / User ───────────────────────────────────────────────────────────────

class TokenResponse(_Base):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    role: UserRole


class UserRead(_Base):
    id: UUID
    site_id: UUID
    email: str | None
    worker_id: str | None
    role: UserRole
    is_active: bool
    created_at: datetime


# ── Near-Miss Event ───────────────────────────────────────────────────────────

class NearMissEventIngest(_Base):
    """
    Posted by Android device on event trigger.
    Audio + frame arrive as multipart, this is the JSON metadata part.
    """
    device_id: str = Field(..., max_length=128)
    worker_id: str = Field(..., max_length=64)   # SHA-256 hash
    site_id: UUID

    event_timestamp: datetime
    gps_lat: float | None = Field(None, ge=-90, le=90)
    gps_lon: float | None = Field(None, ge=-180, le=180)

    # Acoustic
    yamnet_class: str | None = None
    yamnet_class_id: int | None = None
    yamnet_confidence: float | None = Field(None, ge=0.0, le=1.0)
    anomaly_score: float | None = Field(None, ge=0.0, le=1.0)

    # IMU
    imu_jerk_magnitude: float | None = Field(None, ge=0.0)
    imu_timestamp_delta_ms: int | None = None

    # Visual (on-device classification, server will recheck)
    visual_class: str | None = None
    visual_confidence: float | None = Field(None, ge=0.0, le=1.0)


class NearMissEventRead(_Base):
    id: UUID
    site_id: UUID
    event_timestamp: datetime
    received_at: datetime
    status: EventStatus

    gps_lat: float | None
    gps_lon: float | None

    yamnet_class: str | None
    yamnet_confidence: float | None
    anomaly_score: float | None
    imu_jerk_magnitude: float | None

    visual_class: str | None
    visual_confidence: float | None
    frame_description: str | None

    # Presigned URLs (generated on fetch, not stored)
    audio_url: str | None = None
    frame_url: str | None = None

    osha_category: OshaCategory | None
    severity: SeverityLevel | None
    review_notes: str | None
    reviewed_at: datetime | None


class NearMissEventReview(_Base):
    """Supervisor confirms or dismisses an event."""
    status: EventStatus  # CONFIRMED or DISMISSED
    osha_category: OshaCategory | None = None
    severity: SeverityLevel | None = None
    review_notes: str | None = Field(None, max_length=1000)

    @field_validator("status")
    @classmethod
    def must_be_terminal(cls, v: EventStatus) -> EventStatus:
        if v == EventStatus.PENDING:
            raise ValueError("Review must set CONFIRMED or DISMISSED, not PENDING")
        return v


class EventListResponse(_Base):
    items: list[NearMissEventRead]
    total: int
    page: int
    page_size: int


# ── Training Samples ──────────────────────────────────────────────────────────

class TrainingSampleRead(_Base):
    id: UUID
    site_id: UUID
    event_id: UUID
    label_acoustic: str | None
    label_visual: str | None
    label_osha: OshaCategory | None
    label_severity: SeverityLevel | None
    is_used_in_training: bool
    last_model_confidence: float | None
    created_at: datetime


# ── Model Versions ────────────────────────────────────────────────────────────

class ModelVersionRead(_Base):
    id: UUID
    site_id: UUID
    model_type: ModelType
    status: ModelStatus
    version: int
    version_tag: str | None
    val_accuracy: float | None
    previous_val_accuracy: float | None
    per_class_accuracy: dict[str, float] | None
    n_training_samples: int | None
    s3_key: str | None
    trained_at: datetime | None
    deployed_at: datetime | None
    created_at: datetime

    # Generated on response
    download_url: str | None = None


class ModelVersionListResponse(_Base):
    items: list[ModelVersionRead]
    active_acoustic: ModelVersionRead | None
    active_visual: ModelVersionRead | None


# ── Training Jobs ─────────────────────────────────────────────────────────────

class TrainingJobTrigger(_Base):
    """Admin manually triggers a training job."""
    site_id: UUID
    model_type: ModelType
    replay_strategy: str = Field(
        "class_balanced",
        pattern="^(random|class_balanced|uncertainty_weighted)$"
    )


class TrainingJobRead(_Base):
    id: UUID
    site_id: UUID
    model_type: ModelType
    status: TrainingJobStatus
    trigger: str
    n_new_samples: int | None
    n_historical_samples: int | None
    replay_strategy: str
    queued_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    training_log: list[dict[str, Any]] | None  # [{epoch, loss, val_acc}, ...]


# ── Documents / RAG ───────────────────────────────────────────────────────────

class SiteDocumentCreate(_Base):
    site_id: UUID
    doc_type: DocumentType
    # filename set by server from upload


class SiteDocumentRead(_Base):
    id: UUID
    site_id: UUID
    filename: str
    original_filename: str
    doc_type: DocumentType
    status: DocumentStatus
    n_chunks: int | None
    n_tokens_total: int | None
    is_active: bool
    created_at: datetime
    indexed_at: datetime | None
    ingestion_error: str | None


# ── Voice / RAG Query ─────────────────────────────────────────────────────────

class VoiceQueryRequest(_Base):
    site_id: UUID
    query_text: str = Field(..., min_length=1, max_length=500)
    worker_id: str | None = None
    log_query: bool = False  # worker opt-in for logging


class SourceChunk(_Base):
    document_id: str
    document_filename: str
    chunk_text: str
    score: float
    page_number: int | None


class VoiceQueryResponse(_Base):
    answer: str
    detected_intent: DocumentType | None
    sources: list[SourceChunk]
    was_answered: bool
    latency_ms: int


# ── Admin Training Images ─────────────────────────────────────────────────────

class AdminImageRead(_Base):
    id: UUID
    site_id: UUID
    filename: str
    class_label: str
    is_used_in_training: bool
    created_at: datetime


class AdminTrainingStatus(_Base):
    site_id: UUID
    class_label: str
    n_images: int
    is_ready: bool  # n_images >= 15
    last_trained_at: datetime | None
    latest_job: TrainingJobRead | None


# ── Auth ─────────────────────────────────────────────────────────────────────

class SupervisorLoginRequest(_Base):
    email: str
    password: str

class SupervisorRegisterRequest(_Base):
    site_id: UUID
    email: str
    password: str = Field(..., min_length=8)

class WorkerLoginRequest(_Base):
    site_id: UUID
    raw_worker_id: str = Field(..., min_length=1)
    device_id: str = Field(..., min_length=1)

class ChangePasswordRequest(_Base):
    old_password: str
    new_password: str = Field(..., min_length=8)

class SetRoleRequest(_Base):
    target_user_id: UUID
    new_role: UserRole


# ── Site ─────────────────────────────────────────────────────────────────────

class SiteZoneCreate(_Base):
    name: str = Field(..., min_length=1, max_length=255)
    risk_weight: float = Field(1.0, ge=0.1, le=10.0)
    polygon_geojson: dict | None = None

class SiteZoneRead(_Base):
    id: UUID
    site_id: UUID
    name: str
    risk_weight: float
    polygon_geojson: dict | None
    created_at: datetime


# ── Training ─────────────────────────────────────────────────────────────────

class TrainingReadinessResponse(_Base):
    acoustic: dict[str, Any]
    visual: dict[str, Any]

class SampleCountsResponse(_Base):
    total: int
    unused_new: int
    min_required_for_trigger: int
    ready_to_trigger: bool
    by_acoustic_class: dict[str, int]

class ConfidenceUpdateRequest(_Base):
    """Batch update model confidence scores for uncertainty-weighted replay."""
    confidences: dict[str, float]  # {sample_id: confidence_score}

class AdminTrainingTriggerRequest(_Base):
    class_label: str = Field(..., min_length=2, max_length=128)


# ── Model ─────────────────────────────────────────────────────────────────────

class PostDeployAccuracyReport(_Base):
    """Device reports accuracy after deploying new model."""
    site_id: UUID
    model_type: ModelType
    version: int
    reported_accuracy: float = Field(..., ge=0.0, le=1.0)

class ManualRollbackRequest(_Base):
    target_version_id: UUID

class ModelUpdateCheckRequest(_Base):
    site_id: UUID
    model_type: ModelType
    device_current_version: int | None = None


# ── Admin Images ──────────────────────────────────────────────────────────────

class AdminImageClassSummary(_Base):
    class_label: str
    n_images: int
    is_ready: bool
    shortfall: int
    last_uploaded_at: datetime | None
    last_trained_at: datetime | None
    has_new_since_training: bool

class AdminImageUploadResponse(_Base):
    id: UUID
    site_id: UUID
    filename: str
    class_label: str
    file_size_bytes: int | None
    is_used_in_training: bool
    created_at: datetime
    class_total_images: int
    class_is_ready: bool


# ── Analytics ─────────────────────────────────────────────────────────────────

class HeatmapPoint(_Base):
    event_id: UUID
    lat: float
    lon: float
    timestamp: datetime
    status: str
    yamnet_class: str | None
    anomaly_score: float | None
    osha_category: str | None
    severity: str | None
    weight: float

class TrendDataPoint(_Base):
    period: str
    total: int
    confirmed: int
    dismissed: int
    pending: int

class ZoneRiskEntry(_Base):
    zone_id: UUID | None
    zone_name: str
    event_count: int
    avg_anomaly_score: float
    severity_score: int
    risk_level: str

class DashboardSummary(_Base):
    pending_review: int
    oldest_pending_at: datetime | None
    last_7_days: dict[str, Any]
    last_30_days: dict[str, Any]
    generated_at: datetime

class OshaBreakdown(_Base):
    by_osha_category: list[dict[str, Any]]
    by_severity: list[dict[str, Any]]
    top_acoustic_triggers: list[dict[str, Any]]
    period_days: int

class TrainingCoverageResponse(_Base):
    total_events: int
    confirmed: int
    dismissed: int
    pending: int
    review_rate: float
    confirmation_rate: float
    false_positive_rate: float
    training_samples: dict[str, int]


# ── Risk scoring ──────────────────────────────────────────────────────────────

class RiskScoreResponse(_Base):
    event_id: UUID
    risk_score: float
    components: dict[str, Any]


# ── YAMNet recheck ────────────────────────────────────────────────────────────

class YAMNetRecheckResult(_Base):
    event_id: UUID
    yamnet_class: str | None
    yamnet_confidence: float | None
    anomaly_score: float | None
    exceeds_threshold: bool
    top_5: list[dict[str, Any]]


# ── WebSocket ─────────────────────────────────────────────────────────────────

class WSEventMessage(_Base):
    """Pushed to supervisor dashboard WebSocket on new event."""
    type: str = "new_event"
    event_id: UUID
    site_id: UUID
    event_timestamp: datetime
    yamnet_class: str | None
    anomaly_score: float | None
    severity: str | None
    gps_lat: float | None
    gps_lon: float | None


# ── Paginated list helper ─────────────────────────────────────────────────────

class PaginatedResponse(_Base):
    items: list[Any]
    total: int
    page: int
    page_size: int
    has_next: bool

    @classmethod
    def build(
        cls,
        items: list,
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse":
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            has_next=(page * page_size) < total,
        )


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(_Base):
    status: str
    db: str
    qdrant: str
    redis: str
    version: str = "1.0.0"
