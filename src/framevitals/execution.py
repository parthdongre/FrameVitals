"""Adaptive execution budgets for FrameVitals analyses.

The budget layer is intentionally backend-agnostic. It describes how much raw
work an analysis may attempt for a given dataset shape and analysis mode. Native
Rust, Arrow, CUDA, and distributed backends can consume the same contract later.

The immediate goal is safety: expensive statistical routines must never infer
that a 100k-row input means they should allocate O(n^2) intermediates. Sampling
is deterministic and must be disclosed in result metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


_VALID_MODES = {"quick", "standard", "deep", "research"}
_SAMPLE_SEED = 0x9E3779B97F4A7C15

# Full-stream profiling is valuable, but on ultra-wide sources scanning every
# cell defeats the purpose of streaming. These budgets cap the number of source
# cells inspected by the reusable profile pass while preserving the true source
# shape in execution metadata.
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


def _deterministic_stratified_positions(
    rows: int,
    target_rows: int,
    *,
    seed: int = _SAMPLE_SEED,
) -> np.ndarray:
    """Choose one deterministic pseudo-random row from each equal-width stratum.

    Fixed evenly spaced samples can lock onto periodic structure. Stratified jitter
    retains deterministic whole-dataset coverage while breaking that phase locking.
    Positions are returned sorted, so temporal order remains available to callers
    that need it, without allocating a permutation proportional to the source size.
    """
    rows = int(rows)
    target_rows = int(target_rows)
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

    # SplitMix64-style deterministic mixing. uint64 overflow is intentional.
    with np.errstate(over="ignore"):
        mixed = indices + np.uint64(seed)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed = mixed ^ (mixed >> np.uint64(31))

    offsets = (mixed % widths).astype(np.int64)
    positions = edges[:-1] + offsets

    # Keep full-range coverage as an explicit invariant while jittering the
    # interior strata. This is useful for ordered/time-series diagnostics too.
    positions[0] = 0
    positions[-1] = rows - 1
    return positions


def derive_streaming_profile_column_limit(
    rows: int,
    columns: int,
    *,
    mode: str = "standard",
) -> int:
    """Return the deterministic full-stream column budget for a source shape.

    Ordinary datasets keep every column. Ultra-wide/high-cell-count sources are
    projected before the full streaming profile pass so total scanned cells stay
    bounded. The projection itself is selected by the caller from the source
    schema; this function only resolves the allowed width.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown analysis mode: {mode}")
    if rows < 0 or columns < 0:
        raise ValueError("rows and columns must be non-negative.")
    if columns == 0:
        return 0
    if rows == 0:
        return int(columns)

    cells = int(rows) * int(columns)
    cell_budget = int(_STREAMING_PROFILE_CELL_BUDGETS[mode])
    if cells <= cell_budget and columns < 10_000:
        return int(columns)

    by_cells = max(1, cell_budget // max(int(rows), 1))
    return max(
        1,
        min(
            int(columns),
            int(by_cells),
            int(_STREAMING_PROFILE_COLUMN_CAPS[mode]),
        ),
    )


def derive_execution_budget(
    rows: int,
    columns: int,
    *,
    mode: str = "standard",
) -> ExecutionBudget:
    """Derive a conservative execution policy from shape and analysis mode.

    The thresholds are deliberately simple and deterministic for now. They are
    a compatibility layer for the future cost-based planner, where RAM, storage
    metadata, native throughput, GPU availability, and user accuracy budgets can
    refine the same object without changing analysis APIs.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"Unknown analysis mode: {mode}")
    if rows < 0 or columns < 0:
        raise ValueError("rows and columns must be non-negative.")

    cells = int(rows) * int(columns)
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

    # Ultra-wide data must spend relationship budget more carefully. The future
    # sparse feature-graph engine will replace this fixed cap with candidate
    # generation rather than dense pair enumeration.
    relationship_budget = int(selected["relationships"])
    if ultra_wide_dataset:
        relationship_budget = min(relationship_budget, 10)
    elif wide_dataset:
        relationship_budget = min(relationship_budget, 20)

    # Memory-heavy modules should not be launched four-at-a-time simply because
    # a machine exposes four Python workers. Large inputs default to sequential
    # heavy execution until the scheduler gains RAM-aware token accounting.
    heavy_parallelism = 1 if large_dataset or wide_dataset else 2

    return ExecutionBudget(
        mode=mode,
        rows=int(rows),
        columns=int(columns),
        cells=cells,
        scale_class=scale_class,
        large_dataset=large_dataset,
        wide_dataset=wide_dataset,
        ultra_wide_dataset=ultra_wide_dataset,
        quality_sample_rows=_bounded(selected["quality"], rows),
        deep_statistics_sample_rows=_bounded(selected["deep"], rows),
        bootstrap_sample_rows=_bounded(selected["bootstrap"], rows),
        distribution_sample_rows=_bounded(selected["distribution"], rows),
        pair_sample_rows=_bounded(selected["pair"], rows),
        anomaly_sample_rows=_bounded(selected["anomaly"], rows),
        time_series_sample_rows=_bounded(selected["time_series"], rows),
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
    source_rows = int(len(dataframe))
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1.")

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
        # Positions stay sorted so time-aware callers can preserve ordering, while
        # statistical callers receive a copy that is safe to mutate downstream.
        sampled = sampled.copy()

    return sampled, {
        "sampled": True,
        "source_rows": source_rows,
        "sample_rows": int(len(sampled)),
        "strategy": "deterministic_stratified_jitter",
        "preserve_order": bool(preserve_order),
        "seed": int(_SAMPLE_SEED),
    }
