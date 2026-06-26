"""
SentinelSite — Intent Classifier
DistilBERT fine-tuned on 8 construction document categories.
Runs server-side to route voice queries to the right doc type filter.
Falls back to GENERAL if below confidence threshold.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from app.config import settings
from app.db.models import DocumentType

log = logging.getLogger(__name__)

# ── Label map ────────────────────────────────────────────────────────────────

INTENT_LABELS: list[str] = [
    "structural",
    "safety",
    "schedule",
    "material",
    "electrical",
    "plumbing",
    "inspection",
    "general",
]

# Maps label string → DocumentType enum
LABEL_TO_DOC_TYPE: dict[str, DocumentType] = {
    "structural": DocumentType.STRUCTURAL,
    "safety": DocumentType.SAFETY,
    "schedule": DocumentType.SCHEDULE,
    "material": DocumentType.MATERIAL,
    "electrical": DocumentType.ELECTRICAL,
    "plumbing": DocumentType.PLUMBING,
    "inspection": DocumentType.INSPECTION,
    "general": DocumentType.GENERAL,
}

# Confidence threshold — below this, classify as GENERAL (search all docs)
CONFIDENCE_THRESHOLD = 0.55


# ── Result type ───────────────────────────────────────────────────────────────

class IntentResult(NamedTuple):
    doc_type: DocumentType
    label: str
    confidence: float
    all_scores: dict[str, float]


# ── Classifier ────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Lazy-loaded DistilBERT classifier.
    Model is loaded on first call, not at import time (saves memory).
    Falls back to keyword heuristics if model not available.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_path = Path(settings.TFLITE_MODEL_DIR) / "intent_classifier"

    def _load(self) -> None:
        """Load HuggingFace pipeline. Called once."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            model_id = (
                str(self._model_path)
                if self._model_path.exists()
                else "distilbert-base-uncased"  # base model as fallback
            )
            self._pipeline = pipeline(
                task="text-classification",
                model=model_id,
                top_k=None,       # return all class scores
                truncation=True,
                max_length=128,
            )
            log.info(f"Intent classifier loaded from: {model_id}")
        except ImportError:
            log.warning("transformers not installed — using keyword heuristic classifier")
            self._pipeline = "keyword"

    def classify(self, query: str) -> IntentResult:
        """
        Classify query into one of 8 document types.
        Returns IntentResult with predicted type, confidence, and all scores.
        """
        self._load()

        if self._pipeline == "keyword":
            return self._keyword_fallback(query)

        outputs = self._pipeline(query)
        # outputs: [{"label": "LABEL_0", "score": 0.9}, ...]
        scores: dict[str, float] = {}
        for item in outputs[0]:
            # Map LABEL_N → intent label
            idx = int(item["label"].split("_")[-1])
            if idx < len(INTENT_LABELS):
                scores[INTENT_LABELS[idx]] = item["score"]

        if not scores:
            return self._keyword_fallback(query)

        best_label = max(scores, key=lambda k: scores[k])
        best_score = scores[best_label]

        # Below threshold → fall back to GENERAL (no filter = search all)
        if best_score < CONFIDENCE_THRESHOLD:
            best_label = "general"
            best_score = scores.get("general", 1.0 / len(INTENT_LABELS))

        return IntentResult(
            doc_type=LABEL_TO_DOC_TYPE[best_label],
            label=best_label,
            confidence=best_score,
            all_scores=scores,
        )

    def _keyword_fallback(self, query: str) -> IntentResult:
        """
        Rule-based fallback when model not available.
        Good enough for most construction queries.
        """
        q = query.lower()

        rules: list[tuple[str, list[str]]] = [
            ("safety", ["hazard", "ppe", "helmet", "harness", "safety", "risk", "emergency", "osha", "fall protection"]),
            ("structural", ["load", "beam", "column", "foundation", "concrete", "rebar", "structural", "slab", "footing"]),
            ("electrical", ["wire", "circuit", "voltage", "panel", "conduit", "breaker", "electrical", "outlet", "grounding"]),
            ("plumbing", ["pipe", "valve", "drain", "water", "sewage", "plumbing", "fixture", "pressure"]),
            ("schedule", ["schedule", "deadline", "timeline", "milestone", "when", "date", "start", "finish", "phase"]),
            ("material", ["material", "quantity", "order", "supply", "stock", "specification", "grade", "type"]),
            ("inspection", ["inspect", "checklist", "compliance", "code", "permit", "approve", "review", "audit"]),
        ]

        scores: dict[str, float] = {label: 0.0 for label in INTENT_LABELS}
        for label, keywords in rules:
            for kw in keywords:
                if kw in q:
                    scores[label] += 1.0

        total = sum(scores.values())
        if total == 0:
            # No keyword match → GENERAL
            scores["general"] = 1.0
            return IntentResult(
                doc_type=DocumentType.GENERAL,
                label="general",
                confidence=0.5,
                all_scores=scores,
            )

        # Normalize
        scores = {k: v / total for k, v in scores.items()}
        best_label = max(scores, key=lambda k: scores[k])

        if scores[best_label] < CONFIDENCE_THRESHOLD:
            best_label = "general"

        return IntentResult(
            doc_type=LABEL_TO_DOC_TYPE[best_label],
            label=best_label,
            confidence=scores[best_label],
            all_scores=scores,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
intent_classifier = IntentClassifier()
