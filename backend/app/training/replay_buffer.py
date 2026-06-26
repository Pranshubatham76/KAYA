"""
SentinelSite — Experience Replay Buffer
Prevents catastrophic forgetting in continual learning.
Always mixes 70% historical + 30% new samples.
Strategies: random | class_balanced | uncertainty_weighted
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import TrainingSample

log = logging.getLogger(__name__)

ReplayStrategy = Literal["random", "class_balanced", "uncertainty_weighted"]


# ── Sample DTO ────────────────────────────────────────────────────────────────

@dataclass
class ReplaySample:
    """Lightweight sample DTO passed to trainers."""
    id: str
    site_id: str
    audio_s3_key: str | None
    frame_s3_key: str | None
    label_acoustic: str | None
    label_visual: str | None
    label_osha: str | None
    label_severity: str | None
    is_new: bool                        # True = part of the new batch
    model_confidence: float | None      # For uncertainty weighting


# ── Buffer ────────────────────────────────────────────────────────────────────

class ExperienceReplayBuffer:
    """
    Builds training batches by mixing historical and new samples.

    Usage:
        buffer = ExperienceReplayBuffer(db_session)
        batch = buffer.build_batch(
            site_id=site_id,
            new_sample_ids=[...],
            strategy="class_balanced",
            target_total=None,  # auto-compute from ratio
        )
    """

    # FR-L03: 70% historical, 30% new — immutable
    HISTORICAL_RATIO: float = settings.TRAINING_REPLAY_RATIO_HISTORICAL
    NEW_RATIO: float = 1.0 - settings.TRAINING_REPLAY_RATIO_HISTORICAL

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Public ────────────────────────────────────────────────────────────────

    def build_batch(
        self,
        site_id: str | UUID,
        new_sample_ids: list[str],
        strategy: ReplayStrategy = "class_balanced",
        target_total: int | None = None,
    ) -> list[ReplaySample]:
        """
        Build a mixed training batch.

        Args:
            site_id: Current site (historical pulled from same site)
            new_sample_ids: IDs of recently confirmed training samples
            strategy: Sampling strategy for historical pool
            target_total: Total batch size. Defaults to
                          len(new_sample_ids) / NEW_RATIO (ratio-derived)

        Returns:
            Mixed list of ReplaySample, shuffled.
        """
        new_samples = self._fetch_samples(new_sample_ids)
        n_new = len(new_samples)

        if n_new == 0:
            log.warning("No new samples provided to replay buffer")
            return []

        # Compute target sizes from ratio
        if target_total is None:
            # n_new should be 30% of total → total = n_new / 0.30
            target_total = max(n_new, int(n_new / self.NEW_RATIO))

        n_historical_target = int(target_total * self.HISTORICAL_RATIO)

        log.info(
            f"Building replay batch: {n_new} new + {n_historical_target} historical "
            f"(strategy={strategy}, target_total={target_total})"
        )

        # Fetch historical pool (exclude the new sample ids)
        historical_pool = self._fetch_historical_pool(
            site_id=str(site_id),
            exclude_ids=set(new_sample_ids),
        )

        if not historical_pool:
            log.warning(
                f"No historical samples available for site={site_id}. "
                "Training on new samples only — forgetting risk increased."
            )
            result = [self._to_replay(s, is_new=True) for s in new_samples]
            random.shuffle(result)
            return result

        # Sample historical pool with chosen strategy
        historical_selected = self._sample_historical(
            pool=historical_pool,
            n=n_historical_target,
            strategy=strategy,
        )

        # Mix and shuffle
        batch = (
            [self._to_replay(s, is_new=True) for s in new_samples]
            + [self._to_replay(s, is_new=False) for s in historical_selected]
        )
        random.shuffle(batch)

        log.info(
            f"Replay batch ready: {len(batch)} samples "
            f"({n_new} new + {len(historical_selected)} historical)"
        )
        self._log_class_distribution(batch)
        return batch

    def count_available_historical(
        self,
        site_id: str | UUID,
        exclude_ids: set[str] | None = None,
    ) -> int:
        """How many historical samples are available for this site."""
        pool = self._fetch_historical_pool(str(site_id), exclude_ids or set())
        return len(pool)

    # ── Strategies ────────────────────────────────────────────────────────────

    def _sample_historical(
        self,
        pool: list[TrainingSample],
        n: int,
        strategy: ReplayStrategy,
    ) -> list[TrainingSample]:
        """Select n samples from pool using the given strategy."""
        if len(pool) <= n:
            return pool  # not enough historical data — use all

        if strategy == "random":
            return random.sample(pool, n)

        elif strategy == "class_balanced":
            return self._class_balanced_sample(pool, n)

        elif strategy == "uncertainty_weighted":
            return self._uncertainty_weighted_sample(pool, n)

        else:
            log.warning(f"Unknown strategy: {strategy}, using random")
            return random.sample(pool, n)

    def _class_balanced_sample(
        self,
        pool: list[TrainingSample],
        n: int,
    ) -> list[TrainingSample]:
        """
        Equal number of samples per class.
        Critical for preserving rare event classes.
        Uses label_acoustic as primary grouping key.
        """
        by_class: dict[str, list[TrainingSample]] = defaultdict(list)
        for s in pool:
            key = s.label_acoustic or s.label_visual or "unknown"
            by_class[key].append(s)

        n_classes = len(by_class)
        if n_classes == 0:
            return random.sample(pool, min(n, len(pool)))

        per_class = max(1, n // n_classes)
        selected: list[TrainingSample] = []

        for cls, samples in by_class.items():
            chosen = random.sample(samples, min(per_class, len(samples)))
            selected.extend(chosen)

        # If we got fewer than n (classes with <per_class samples), top up randomly
        if len(selected) < n:
            remaining_pool = [s for s in pool if s not in set(selected)]
            extra = min(n - len(selected), len(remaining_pool))
            if extra > 0:
                selected.extend(random.sample(remaining_pool, extra))

        return selected[:n]

    def _uncertainty_weighted_sample(
        self,
        pool: list[TrainingSample],
        n: int,
    ) -> list[TrainingSample]:
        """
        Prioritize samples the current model was least confident about.
        Low confidence = high uncertainty = high weight.
        Falls back to random if no confidence scores available.
        """
        with_confidence = [s for s in pool if s.last_model_confidence is not None]

        if len(with_confidence) < n // 2:
            log.debug("Insufficient confidence scores — mixing uncertainty + random")
            # Half uncertainty, half random from full pool
            uncertainty_n = len(with_confidence)
            random_n = n - uncertainty_n
            uncertainty_selected = with_confidence  # all of them
            random_pool = [s for s in pool if s not in set(with_confidence)]
            random_selected = random.sample(random_pool, min(random_n, len(random_pool)))
            return uncertainty_selected + random_selected

        # Weight = 1 - confidence (lower confidence → higher weight)
        weights = [1.0 - (s.last_model_confidence or 0.5) for s in with_confidence]
        total_w = sum(weights) or 1.0
        probs = [w / total_w for w in weights]

        # Weighted sampling without replacement
        import numpy as np
        indices = np.random.choice(
            len(with_confidence), size=min(n, len(with_confidence)), replace=False, p=probs
        )
        return [with_confidence[i] for i in indices]

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _fetch_samples(self, ids: list[str]) -> list[TrainingSample]:
        if not ids:
            return []
        return (
            self._db.query(TrainingSample)
            .filter(TrainingSample.id.in_(ids))
            .all()
        )

    def _fetch_historical_pool(
        self,
        site_id: str,
        exclude_ids: set[str],
    ) -> list[TrainingSample]:
        """
        All confirmed training samples for site, excluding the current new batch.
        No cap — let strategy decide how many to use.
        """
        q = (
            self._db.query(TrainingSample)
            .filter(
                TrainingSample.site_id == site_id,
                TrainingSample.is_used_in_training == True,
            )
        )
        if exclude_ids:
            q = q.filter(TrainingSample.id.notin_(exclude_ids))

        return q.order_by(TrainingSample.created_at.desc()).all()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_replay(sample: TrainingSample, is_new: bool) -> ReplaySample:
        return ReplaySample(
            id=str(sample.id),
            site_id=str(sample.site_id),
            audio_s3_key=sample.audio_s3_key,
            frame_s3_key=sample.frame_s3_key,
            label_acoustic=sample.label_acoustic,
            label_visual=sample.label_visual,
            label_osha=sample.label_osha.value if sample.label_osha else None,
            label_severity=sample.label_severity.value if sample.label_severity else None,
            is_new=is_new,
            model_confidence=sample.last_model_confidence,
        )

    def _log_class_distribution(self, batch: list[ReplaySample]) -> None:
        """Log class distribution for debugging and monitoring."""
        by_class: dict[str, int] = defaultdict(int)
        for s in batch:
            key = s.label_acoustic or s.label_visual or "unknown"
            by_class[key] += 1
        log.info(f"Batch class distribution: {dict(sorted(by_class.items()))}")
