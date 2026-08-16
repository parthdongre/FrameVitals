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

    # Evenly spaced positions are deterministic, preserve coverage across the
    # source, and avoid allocating a random permutation proportional to n.
    positions = np.linspace(0, source_rows - 1, num=max_rows, dtype=np.int64)
    positions = np.unique(positions)
    sampled = dataframe.iloc[positions]
    if not preserve_order:
        # The positions are already ordered; this branch documents that callers
        # may treat the sample as an unordered statistical sample.
        sampled = sampled.copy()

    return sampled, {
        "sampled": True,
        "source_rows": source_rows,
        "sample_rows": int(len(sampled)),
        "strategy": "deterministic_evenly_spaced",
        "preserve_order": bool(preserve_order),
    }
