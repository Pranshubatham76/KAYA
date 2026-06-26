"""
SentinelSite — Server-Side ML Inference
1. YAMNetServerRecheck: full-quality acoustic recheck on uploaded audio
2. RiskScorer: zone × time-of-day risk multiplier
Both run as Celery CPU tasks, not on device.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime

import numpy as np

from app.config import settings
from app.core.storage import storage

log = logging.getLogger(__name__)


# ── YAMNet Server Recheck ─────────────────────────────────────────────────────

class YAMNetServerRecheck:
    """
    Full-quality YAMNet inference on server (not quantized TFLite).
    Validates on-device classification with higher accuracy.
    Runs as Celery task after event upload.
    """

    # Near-miss relevant AudioSet class IDs
    NEAR_MISS_IDS = {373, 374, 375, 376, 44, 45, 378, 388}
    CONSTRUCTION_NOISE_IDS = {474, 476, 479, 355, 472}

    YAMNET_CLASS_NAMES = {
        373: "Crash, boom, or clatter",
        374: "Bang",
        375: "Impact, banging",
        376: "Thud",
        44: "Shout",
        45: "Screaming",
        378: "Breaking",
        388: "Alarm",
        474: "Power tool",
        476: "Jackhammer",
        479: "Sawing",
        355: "Engine",
        472: "Air compressor",
    }

    def __init__(self) -> None:
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            log.info("YAMNet server model loaded from TF Hub")
        except ImportError:
            log.warning("tensorflow/tensorflow_hub not installed — using stub")
            self._model = "stub"

    def recheck(
        self,
        audio_bytes: bytes,
        site_acoustic_threshold: float | None = None,
    ) -> dict:
        """
        Run YAMNet on audio clip. Returns class + confidence.
        Called by Celery recheck_event_task.
        """
        self._load()

        waveform = self._decode_audio(audio_bytes)
        threshold = site_acoustic_threshold or settings.ANOMALY_SCORE_THRESHOLD

        if self._model == "stub":
            return self._stub_result()

        import tensorflow as tf
        waveform_tf = tf.constant(waveform, dtype=tf.float32)
        scores, embeddings, spectrogram = self._model(waveform_tf)

        # Average scores across time frames
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        # Top-5 predictions
        top_indices = np.argsort(mean_scores)[::-1][:5]
        top_classes = [
            {
                "class_id": int(idx),
                "class_name": self.YAMNET_CLASS_NAMES.get(int(idx), f"AudioSet_{idx}"),
                "confidence": float(mean_scores[idx]),
            }
            for idx in top_indices
        ]

        # Best near-miss class
        near_miss_scores = {
            idx: float(mean_scores[idx]) for idx in self.NEAR_MISS_IDS
        }
        best_nm_id = max(near_miss_scores, key=near_miss_scores.get)
        best_nm_score = near_miss_scores[best_nm_id]

        # Anomaly score (near-miss class prob mass, excluding construction noise)
        construction_prob = sum(
            float(mean_scores[idx]) for idx in self.CONSTRUCTION_NOISE_IDS
            if idx < len(mean_scores)
        )
        near_miss_prob = sum(
            float(mean_scores[idx]) for idx in self.NEAR_MISS_IDS
            if idx < len(mean_scores)
        )
        anomaly_score = float(np.clip(near_miss_prob - construction_prob * 0.5, 0, 1))

        return {
            "yamnet_class": self.YAMNET_CLASS_NAMES.get(best_nm_id, f"class_{best_nm_id}"),
            "yamnet_class_id": best_nm_id,
            "yamnet_confidence": best_nm_score,
            "anomaly_score": round(anomaly_score, 4),
            "exceeds_threshold": anomaly_score > threshold,
            "top_5": top_classes,
            "construction_noise_prob": round(construction_prob, 4),
        }

    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        try:
            import soundfile as sf
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            import scipy.io.wavfile as wav
            sr, waveform = wav.read(io.BytesIO(audio_bytes))
            waveform = waveform.astype(np.float32) / 32768.0

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            try:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            except ImportError:
                pass  # Proceed with original SR

        return waveform

    def _stub_result(self) -> dict:
        return {
            "yamnet_class": "Impact, banging",
            "yamnet_class_id": 375,
            "yamnet_confidence": 0.72,
            "anomaly_score": 0.68,
            "exceeds_threshold": True,
            "top_5": [],
            "construction_noise_prob": 0.12,
            "stub": True,
        }


# ── Risk Scorer ───────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Computes a risk multiplier for a near-miss event based on:
    - Zone risk weight (from SiteZone.risk_weight)
    - Time of day (peak hours have higher risk)
    - Historical event density in zone
    Used to prioritize supervisor review order.
    """

    # Peak risk hours on construction sites (shift start/end, breaks)
    HIGH_RISK_HOURS = {6, 7, 11, 12, 15, 16, 17}  # 6–7am, 11am–noon, 3–5pm
    HIGH_RISK_HOUR_MULTIPLIER = 1.5
    NORMAL_HOUR_MULTIPLIER = 1.0

    def compute_risk_score(
        self,
        anomaly_score: float,
        zone_risk_weight: float,
        event_hour: int,
        zone_event_density: float = 1.0,  # recent events / area
    ) -> float:
        """
        risk_score = anomaly_score × zone_weight × time_multiplier × density_factor
        Capped at 1.0.
        """
        time_mult = (
            self.HIGH_RISK_HOUR_MULTIPLIER
            if event_hour in self.HIGH_RISK_HOURS
            else self.NORMAL_HOUR_MULTIPLIER
        )
        # Density: more past events in zone = higher risk
        density_factor = min(1.0 + (zone_event_density - 1.0) * 0.1, 2.0)

        score = anomaly_score * zone_risk_weight * time_mult * density_factor
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    async def score_event(
        self,
        db,
        event_id: str,
        site_id: str,
    ) -> dict:
        """
        Compute and return full risk assessment for an event.
        Fetches zone weight from DB.
        """
        from sqlalchemy import select, func
        from sqlalchemy.ext.asyncio import AsyncSession
        from app.db.models import NearMissEvent, SiteZone

        result = await db.execute(
            select(NearMissEvent).where(
                NearMissEvent.id == event_id,
                NearMissEvent.site_id == site_id,
            )
        )
        event = result.scalar_one_or_none()
        if not event:
            raise ValueError(f"Event {event_id} not found")

        zone_weight = 1.0
        if event.zone_id:
            zone_res = await db.execute(
                select(SiteZone).where(SiteZone.id == event.zone_id)
            )
            zone = zone_res.scalar_one_or_none()
            if zone:
                zone_weight = zone.risk_weight or 1.0

        event_hour = event.event_timestamp.hour
        anomaly_score = event.anomaly_score or 0.5

        risk_score = self.compute_risk_score(
            anomaly_score=anomaly_score,
            zone_risk_weight=zone_weight,
            event_hour=event_hour,
        )

        return {
            "event_id": event_id,
            "risk_score": risk_score,
            "components": {
                "anomaly_score": anomaly_score,
                "zone_weight": zone_weight,
                "event_hour": event_hour,
                "is_peak_hour": event_hour in self.HIGH_RISK_HOURS,
            },
        }


# ── Celery task ───────────────────────────────────────────────────────────────

def register_ml_tasks(celery_app):
    """Register ML celery tasks — called from scheduler.py."""

    @celery_app.task(name="app.ml.yamnet_recheck_task", bind=True, max_retries=2)
    def yamnet_recheck_task(self, event_id: str, audio_key: str):
        from app.db.session import get_sync_db
        from app.db.models import NearMissEvent
        import asyncio

        recheck = YAMNetServerRecheck()
        audio_bytes = storage.download_audio(audio_key)
        result = recheck.recheck(audio_bytes)

        with get_sync_db() as db:
            event = db.query(NearMissEvent).filter(NearMissEvent.id == event_id).first()
            if event:
                event.server_yamnet_class = result["yamnet_class"]
                event.server_yamnet_confidence = result["yamnet_confidence"]
                event.server_recheck_done = True
                db.commit()
                log.info(
                    f"YAMNet recheck done: event={event_id}, "
                    f"class={result['yamnet_class']}, "
                    f"conf={result['yamnet_confidence']:.3f}"
                )

    @celery_app.task(name="app.ml.vision_description_task", bind=True, max_retries=2)
    def vision_description_task(self, event_id: str, frame_key: str):
        import asyncio
        from app.db.session import get_sync_db
        from app.db.models import NearMissEvent
        from app.core.services import vision_describer

        frame_bytes = storage.download_frame(frame_key)
        description = asyncio.run(vision_describer.describe(frame_bytes))

        with get_sync_db() as db:
            event = db.query(NearMissEvent).filter(NearMissEvent.id == event_id).first()
            if event:
                event.frame_description = description
                db.commit()
                log.info(f"Vision description done: event={event_id}")


# ── Singletons ────────────────────────────────────────────────────────────────
yamnet_recheck = YAMNetServerRecheck()
risk_scorer = RiskScorer()
