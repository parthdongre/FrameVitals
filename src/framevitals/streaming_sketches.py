"""Pure-NumPy bounded-memory sketches for non-native streaming execution.

The Rust backend already owns the fastest full-stream sketch implementation.
These fallbacks mirror the same logarithmic-quantile semantics for datasets
small enough that Python/NumPy sketch maintenance is cheaper than retaining a
large row sample. Ultra-wide inputs deliberately keep the existing sampled
fallback so this module never turns a memory fix into a CPU regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


PYTHON_NUMERIC_SKETCH_CELL_BUDGET = 50_000_000
DEFAULT_RELATIVE_ACCURACY = 0.01
DEFAULT_ZERO_THRESHOLD = 1.0e-12


def should_use_full_stream_numpy_sketch(
    rows: int,
    numeric_columns: int,
    *,
    cell_budget: int = PYTHON_NUMERIC_SKETCH_CELL_BUDGET,
) -> bool:
    """Return whether NumPy sketch work fits the configured cell-cost budget."""
    if rows < 0 or numeric_columns < 0:
        raise ValueError("rows and numeric_columns must be non-negative.")
    if cell_budget < 1:
        raise ValueError("cell_budget must be positive.")
    return int(rows) * int(numeric_columns) <= int(cell_budget)


@dataclass(slots=True)
class NumpyLogQuantileSketch:
    """Mergeable relative-accuracy logarithmic quantile sketch.

    This mirrors FrameVitals' native ``LogQuantileSketch`` rather than retaining
    raw observations. Batch updates are vectorized with NumPy and only unique
    logarithmic bins are folded into Python dictionaries.
    """

    relative_accuracy: float = DEFAULT_RELATIVE_ACCURACY
    zero_threshold: float = DEFAULT_ZERO_THRESHOLD
    negative: dict[int, int] = field(default_factory=dict)
    positive: dict[int, int] = field(default_factory=dict)
    zero_count: int = 0
    count: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.relative_accuracy) < 1.0:
            raise ValueError("relative_accuracy must be between 0 and 1.")
        if self.zero_threshold < 0:
            raise ValueError("zero_threshold must be non-negative.")

    @property
    def _log_gamma(self) -> float:
        gamma = (1.0 + self.relative_accuracy) / (1.0 - self.relative_accuracy)
        return float(np.log(gamma))

    def _update_side(self, target: dict[int, int], magnitudes: np.ndarray) -> None:
        if magnitudes.size == 0:
            return
        keys = np.floor(np.log(magnitudes) / self._log_gamma).astype(np.int64)
        unique, counts = np.unique(keys, return_counts=True)
        for key, amount in zip(unique.tolist(), counts.tolist(), strict=True):
            integer_key = int(key)
            target[integer_key] = target.get(integer_key, 0) + int(amount)

    def update(self, values: Any) -> "NumpyLogQuantileSketch":
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return self

        self.count += int(finite.size)
        absolute = np.abs(finite)
        zero_mask = absolute <= self.zero_threshold
        self.zero_count += int(np.count_nonzero(zero_mask))

        nonzero = finite[~zero_mask]
        if nonzero.size:
            negative = nonzero[nonzero < 0.0]
            positive = nonzero[nonzero > 0.0]
            self._update_side(self.negative, -negative)
            self._update_side(self.positive, positive)
        return self

    def merge(self, other: "NumpyLogQuantileSketch") -> "NumpyLogQuantileSketch":
        if not np.isclose(self.relative_accuracy, other.relative_accuracy):
            raise ValueError("quantile sketch accuracy mismatch")
        if not np.isclose(self.zero_threshold, other.zero_threshold):
            raise ValueError("quantile sketch zero-threshold mismatch")
        self.count += other.count
        self.zero_count += other.zero_count
        for key, amount in other.negative.items():
            self.negative[key] = self.negative.get(key, 0) + amount
        for key, amount in other.positive.items():
            self.positive[key] = self.positive.get(key, 0) + amount
        return self

    def _representative(self, key: int) -> float:
        return float(np.exp((float(key) + 0.5) * self._log_gamma))

    def quantile(self, q: float) -> float | None:
        if self.count == 0:
            return None
        if not 0.0 <= float(q) <= 1.0:
            raise ValueError("q must be between 0 and 1.")

        target = int(np.floor(float(q) * (self.count - 1)))
        seen = 0
        for key in sorted(self.negative, reverse=True):
            amount = self.negative[key]
            if target < seen + amount:
                return -self._representative(key)
            seen += amount

        if target < seen + self.zero_count:
            return 0.0
        seen += self.zero_count

        for key in sorted(self.positive):
            amount = self.positive[key]
            if target < seen + amount:
                return self._representative(key)
            seen += amount

        if self.positive:
            return self._representative(max(self.positive))
        if self.negative:
            return -self._representative(min(self.negative))
        return 0.0

    @property
    def bin_count(self) -> int:
        return len(self.negative) + len(self.positive) + int(self.zero_count > 0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "method": "numpy_log_quantile_sketch",
            "count": int(self.count),
            "relative_accuracy": float(self.relative_accuracy),
            "bin_count": int(self.bin_count),
            "p01": self.quantile(0.01),
            "p05": self.quantile(0.05),
            "p25": self.quantile(0.25),
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
        }
