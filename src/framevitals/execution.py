"""Adaptive execution budgets for FrameVitals analyses.

The budget layer is intentionally backend-agnostic. It describes how much raw
work an analysis may attempt for a given dataset shape and analysis mode. Native
Rust, Arrow, CUDA, and distributed backends can consume the same contract later.

The immediate goal is safety: expensive statistical routines must never infer
that a 100k-row input means they should allocate O(n^2) intermediates. Sampling
is deterministic and must be disclosed in result metadata.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Iterator

import numpy as np
import pandas as pd


_VALID_MODES = {"quick", "standard", "deep", "research"}
_SAMPLE_SEED = 0x9E3779B97F4A7C15

_STREAMING_PROFILE_CELL_BUDGETS = {
    "quick": 64_000_000,
    "standard": 96_000_000,
    "deep": 128_000_000,
    "research": 256_000_000,
}
_STREAMING_PROFILE_COLUMN_CAPS = {
    "quick": 64,
    "standard": 96,
    "deep": 128,
    "research": 256,
}


def _require_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative.")
    return converted


def _require_positive_int(name: str, value: Any) -> int:
    converted = _require_non_negative_int(name, value)
    if converted < 1:
        raise ValueError(f"{name} must be at least 1.")
    return converted


def _optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _require_positive_int(name, value)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Per-call hard caps layered on top of adaptive mode defaults.

    Policy values can only reduce work. They never increase a mode's built-in
    sampling or parallelism limits. A context variable carries the policy across
    the existing pipeline without mutable process-wide globals, so concurrent
    analyses can safely use different limits.
    """

    max_sample_rows: int | None = None
    max_relationship_pairs: int | None = None
    max_memory_heavy_parallelism: int | None = None
    max_streaming_profile_columns: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_sample_rows",
            "max_relationship_pairs",
            "max_memory_heavy_parallelism",
            "max_streaming_profile_columns",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_int(name, getattr(self, name)),
            )

    def to_dict(self) -> dict[str, int | None]:
        return asdict(self)


_DEFAULT_EXECUTION_POLICY = ExecutionPolicy()
_EXECUTION_POLICY: ContextVar[ExecutionPolicy] = ContextVar(
    "framevitals_execution_policy",
    default=_DEFAULT_EXECUTION_POLICY,
)


def current_execution_policy() -> ExecutionPolicy:
    """Return the resource caps active for the current analysis context."""
    return _EXECUTION_POLICY.get()


@contextmanager
def use_execution_policy(policy: ExecutionPolicy) -> Iterator[None]:
    """Apply ``policy`` to nested budget derivation for one logical run."""
    if not isinstance(policy, ExecutionPolicy):
        raise TypeError("policy must be an ExecutionPolicy.")
    token = _EXECUTION_POLICY.set(policy)
    try:
        yield
    finally:
        _EXECUTION_POLICY.reset(token)


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Resolved resource policy for one analysis run.

    Counts are upper bounds, not promises that every analysis consumes the full
    allowance. Backends are encouraged to stop earlier when an estimate has
    converged or an operation is not applicable.
    """

    mode: str
    rows: int
    columns: int
    cells: int
    scale_class: str
    large_dataset: bool
    wide_dataset: bool
    ultra_wide_dataset: bool
    quality_sample_rows: int
    deep_statistics_sample_rows: int
    bootstrap_sample_rows: int
    distribution_sample_rows: int
    pair_sample_rows: int
    anomaly_sample_rows: int
    time_series_sample_rows: int
    relationship_pair_budget: int
    max_memory_heavy_parallelism: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded(requested: int, rows: int) -> int:
    if rows <= 0:
        return 0
    return min(int(requested), int(rows))


def _policy_row_cap(requested: int, rows: int, policy: ExecutionPolicy) -> int:
    bounded = _bounded(requested, rows)
    if policy.max_sample_rows is None:
        return bounded
    return min(bounded, policy.max_sample_rows)


def _deterministic_stratified_positions(
    rows: int,
    target_rows: int,
    *,
    seed: int = _SAMPLE_SEED,
) -> np.ndarray:
    """Choose one deterministic pseudo-random row from each equal-width stratum."""
    rows = _require_non_negative_int("rows", rows)
    target_rows = _require_non_negative_int("target_rows", target_rows)
    count = min(rows, target_rows)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == rows:
        return np.arange(rows, dtype=np.int64)
    if count == 1:
        return np.array([rows // 2], dtype=np.int64)

    edges = np.fromiter(
        ((index * rows) // count for index in range(count + 1)),
        dtype=np.int64,
        count=count + 1,
    )
    widths = (edges[1:] - edges[:-1]).astype(np.uint64)
    indices = np.arange(count, dtype=np.uint64)

    with np.errstate(over="ignore"):
        mixed = indices + np.uint64(seed)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed = mixed ^ (mixed >> np.uint64(31))

    offsets = (mixed % widths).astype(np.int64)
    positions = edges[:-1] + offsets
    positions[0] = 0
    positions[-1] = rows - 1
    return positions


def derive_streaming_profile_column_limit(
    rows: int,
    columns: int,
    *,
    mode: str = "standard",
) -> int:
    """Return the deterministic full-stream column budget for a source shape."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown analysis mode: {mode}")
    rows = _require_non_negative_int("rows", rows)
    columns = _require_non_negative_int("columns", columns)
    if columns == 0:
        return 0

    policy = current_execution_policy()
    explicit_cap = policy.max_streaming_profile_columns

    if rows == 0:
        derived = columns
    else:
        cells = rows * columns
        cell_budget = int(_STREAMING_PROFILE_CELL_BUDGETS[mode])
        if cells <= cell_budget and columns < 10_000:
            derived = columns
        else:
            by_cells = max(1, cell_budget // max(rows, 1))
            derived = max(
                1,
                min(
                    columns,
                    int(by_cells),
                    int(_STREAMING_PROFILE_COLUMN_CAPS[mode]),
                ),
            )

    if explicit_cap is not None:
        derived = min(derived, explicit_cap)
    return max(1, int(derived))


def derive_execution_budget(
    rows: int,
    columns: int,
    *,
    mode: str = "standard",
) -> ExecutionBudget:
    """Derive a conservative execution policy from shape and analysis mode."""
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown analysis mode: {mode}")
    rows = _require_non_negative_int("rows", rows)
    columns = _require_non_negative_int("columns", columns)

    policy = current_execution_policy()
    cells = rows * columns
    large_dataset = rows >= 100_000 or cells >= 10_000_000
    wide_dataset = columns >= 1_000
    ultra_wide_dataset = columns >= 10_000

    if rows >= 100_000_000 or columns >= 100_000 or cells >= 1_000_000_000_000:
        scale_class = "extreme"
    elif rows >= 10_000_000 or columns >= 10_000 or cells >= 10_000_000_000:
        scale_class = "very_large"
    elif large_dataset or wide_dataset:
        scale_class = "large"
    else:
        scale_class = "normal"

    presets = {
        "quick": {
            "quality": 1_000,
            "deep": 2_000,
            "bootstrap": 1_000,
            "distribution": 2_000,
            "pair": 2_000,
            "anomaly": 5_000,
            "time_series": 5_000,
            "relationships": 10,
        },
        "standard": {
            "quality": 5_000,
            "deep": 5_000,
            "bootstrap": 2_500,
            "distribution": 5_000,
            "pair": 5_000,
            "anomaly": 10_000,
            "time_series": 10_000,
            "relationships": 20,
        },
        "deep": {
            "quality": 10_000,
            "deep": 10_000,
            "bootstrap": 5_000,
            "distribution": 10_000,
            "pair": 10_000,
            "anomaly": 25_000,
            "time_series": 25_000,
            "relationships": 50,
        },
        "research": {
            "quality": 20_000,
            "deep": 20_000,
            "bootstrap": 10_000,
            "distribution": 20_000,
            "pair": 20_000,
            "anomaly": 50_000,
            "time_series": 50_000,
            "relationships": 100,
        },
    }
    selected = presets[mode]

    relationship_budget = int(selected["relationships"])
    if ultra_wide_dataset:
        relationship_budget = min(relationship_budget, 10)
    elif wide_dataset:
        relationship_budget = min(relationship_budget, 20)
    if policy.max_relationship_pairs is not None:
        relationship_budget = min(
            relationship_budget,
            policy.max_relationship_pairs,
        )

    heavy_parallelism = 1 if large_dataset or wide_dataset else 2
    if policy.max_memory_heavy_parallelism is not None:
        heavy_parallelism = min(
            heavy_parallelism,
            policy.max_memory_heavy_parallelism,
        )

    return ExecutionBudget(
        mode=mode,
        rows=rows,
        columns=columns,
        cells=cells,
        scale_class=scale_class,
        large_dataset=large_dataset,
        wide_dataset=wide_dataset,
        ultra_wide_dataset=ultra_wide_dataset,
        quality_sample_rows=_policy_row_cap(selected["quality"], rows, policy),
        deep_statistics_sample_rows=_policy_row_cap(selected["deep"], rows, policy),
        bootstrap_sample_rows=_policy_row_cap(selected["bootstrap"], rows, policy),
        distribution_sample_rows=_policy_row_cap(
            selected["distribution"], rows, policy
        ),
        pair_sample_rows=_policy_row_cap(selected["pair"], rows, policy),
        anomaly_sample_rows=_policy_row_cap(selected["anomaly"], rows, policy),
        time_series_sample_rows=_policy_row_cap(selected["time_series"], rows, policy),
        relationship_pair_budget=relationship_budget,
        max_memory_heavy_parallelism=heavy_parallelism,
    )


def deterministic_sample_frame(
    dataframe: pd.DataFrame,
    max_rows: int,
    *,
    preserve_order: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a deterministic bounded view and transparent sampling metadata."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    max_rows = _require_positive_int("max_rows", max_rows)
    source_rows = int(len(dataframe))

    if source_rows <= max_rows:
        return dataframe, {
            "sampled": False,
            "source_rows": source_rows,
            "sample_rows": source_rows,
            "strategy": "full",
        }

    positions = _deterministic_stratified_positions(source_rows, max_rows)
    sampled = dataframe.iloc[positions]
    if not preserve_order:
        sampled = sampled.copy()

    return sampled, {
        "sampled": True,
        "source_rows": source_rows,
        "sample_rows": int(len(sampled)),
        "strategy": "deterministic_stratified_jitter",
        "preserve_order": bool(preserve_order),
        "seed": int(_SAMPLE_SEED),
    }
