"""
SentinelSite — Core Services Registry
Single import surface for all services.

Usage in route files:
    from app.core import event_service, auth_service, analytics_service, ...
"""
from app.core.services import (
    event_service,
    document_service,
    voice_service,
    vision_describer,
    site_service,
)
from app.core.auth_service import auth_service
from app.core.training_service import training_service
from app.core.model_service import model_service
from app.core.admin_image_service import admin_image_service
from app.core.analytics_service import analytics_service
from app.core.storage import storage
from app.core.vector_db import vector_db
from app.ml.inference import yamnet_recheck, risk_scorer

__all__ = [
    "event_service",
    "document_service",
    "voice_service",
    "vision_describer",
    "site_service",
    "auth_service",
    "training_service",
    "model_service",
    "admin_image_service",
    "analytics_service",
    "storage",
    "vector_db",
    "yamnet_recheck",
    "risk_scorer",
]
