"""Low-overhead change detection for streaming numeric batches.

The detector consumes one aggregate observation per bounded Arrow batch rather
than every raw cell. That makes its Python overhead proportional to
``columns * batches`` while still surfacing sustained within-file mean shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math


@dataclass(slots=True)
class PageHinkleyMeanShift:
    """Scale-adaptive Page-Hinkley-style detector for sequential batch means."""

    threshold: float = 8.0
    delta: float = 0.05
    min_updates: int = 8
    alpha: float = 0.995
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    cumulative_up: float = 0.0
    cumulative_down: float = 0.0
    min_up: float = 0.0
    min_down: float = 0.0
    max_score: float = 0.0
    detected: bool = False
    direction: str | None = None
    detected_at: int | None = None

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("threshold must be positive.")
        if self.delta < 0:
            raise ValueError("delta must be non-negative.")
        if self.min_updates < 3:
            raise ValueError("min_updates must be at least 3.")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")

    @property
    def variance(self) -> float | None:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float | None:
        variance = self.variance
        if variance is None or variance <= 0:
            return None
        return math.sqrt(variance)

    def update(self, value: float | int | None) -> bool:
        """Update with the next batch mean and return current detection state."""
        if value is None:
            return self.detected
        x = float(value)
        if not math.isfinite(x):
            return self.detected

        previous_mean = self.mean
        previous_std = self.std
        self.count += 1

        if self.count == 1:
            self.mean = x
            return self.detected

        difference = x - self.mean
        self.mean += difference / self.count
        self.m2 += difference * (x - self.mean)

        if self.count < self.min_updates:
            return self.detected

        scale = previous_std if previous_std is not None and previous_std > 1e-12 else None
        if scale is None:
            scale = max(abs(previous_mean) * 1e-6, 1e-9)

        residual = (x - previous_mean) / scale
        up_increment = residual - self.delta
        down_increment = -residual - self.delta

        self.cumulative_up = self.alpha * self.cumulative_up + up_increment
        self.cumulative_down = self.alpha * self.cumulative_down + down_increment
        self.min_up = min(self.min_up, self.cumulative_up)
        self.min_down = min(self.min_down, self.cumulative_down)

        up_score = self.cumulative_up - self.min_up
        down_score = self.cumulative_down - self.min_down
        score = max(up_score, down_score)
        self.max_score = max(self.max_score, float(score))

        if not self.detected and score >= self.threshold:
            self.detected = True
            self.direction = "up" if up_score >= down_score else "down"
            self.detected_at = self.count
        return self.detected

    def snapshot(self) -> dict[str, Any]:
        return {
            "method": "page_hinkley_batch_means",
            "updates": int(self.count),
            "detected": bool(self.detected),
            "direction": self.direction,
            "detected_at_batch": self.detected_at,
            "max_score": round(float(self.max_score), 6),
            "threshold": float(self.threshold),
            "delta": float(self.delta),
            "minimum_batches": int(self.min_updates),
            "sufficient_batches": bool(self.count >= self.min_updates),
        }
