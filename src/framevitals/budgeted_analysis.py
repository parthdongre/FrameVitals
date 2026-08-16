"""Resource-bounded adapters around legacy analysis modules.

These adapters are a migration bridge. They make existing analyses safe on
large inputs today while FrameVitals evolves toward streaming Rust/Arrow kernels
and per-operation exact/approximate execution. Sampling is deterministic and
always included in returned metadata.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from framevitals.anomaly_ensemble import detect_anomalies_ensemble
from framevitals.deep_statistics_v2 import run_deep_statistics_v2
from framevitals.execution import ExecutionBudget, deterministic_sample_frame
from framevitals.time_series import detect_and_analyze_time_series


def _attach_execution(
    payload: dict[str, Any],
    *,
    budget: ExecutionBudget,
    sampling: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    result = dict(payload)
    result["execution"] = {
        "scope": scope,
        "scale_class": budget.scale_class,
        **sampling,
    }
    return result


def run_budgeted_deep_statistics(
    dataframe: pd.DataFrame,
    *,
    budget: ExecutionBudget,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Run legacy deep statistics without exposing it to unbounded row counts.

    The current v2 implementation performs several operations over one shared
    frame, including SciPy BCa bootstrap. Until those operations are separated,
    the strictest relevant row budget is used for the shared frame.
    """
    sample_limit = max(
        20,
        min(
            budget.deep_statistics_sample_rows,
            budget.bootstrap_sample_rows,
        ),
    )
    sample_limit = min(sample_limit, max(len(dataframe), 1))
    work, sampling = deterministic_sample_frame(dataframe, sample_limit)

    pair_budget = (
        budget.relationship_pair_budget
        if max_pairs is None
        else min(int(max_pairs), budget.relationship_pair_budget)
    )
    if pair_budget < 1:
        raise ValueError("max_pairs must be at least 1.")

    payload = run_deep_statistics_v2(work, max_pairs=pair_budget)
    sampling = {
        **sampling,
        "reason": (
            "Legacy deep-statistics operations share one bounded frame; "
            "the strict bootstrap budget is applied to prevent quadratic BCa memory growth."
            if sampling["sampled"]
            else "Full input fits within the deep-statistics execution budget."
        ),
        "pair_budget": int(pair_budget),
    }
    return _attach_execution(
        payload,
        budget=budget,
        sampling=sampling,
        scope="bounded_deep_statistics",
    )


def run_budgeted_anomalies(
    dataframe: pd.DataFrame,
    *,
    budget: ExecutionBudget,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
) -> dict[str, Any]:
    """Run the anomaly ensemble on a bounded representative frame when needed."""
    sample_limit = max(20, min(budget.anomaly_sample_rows, max(len(dataframe), 1)))
    work, sampling = deterministic_sample_frame(dataframe, sample_limit)
    payload = detect_anomalies_ensemble(
        work,
        contamination=contamination,
        threshold=threshold,
        max_columns=max_columns,
        top_k=top_k,
    )
    sampling = {
        **sampling,
        "reason": (
            "Expensive neighbor/covariance detectors are bounded until the native "
            "candidate-filtering anomaly engine is available."
            if sampling["sampled"]
            else "Full input fits within the anomaly execution budget."
        ),
        "coverage": "sample" if sampling["sampled"] else "full",
    }
    return _attach_execution(
        payload,
        budget=budget,
        sampling=sampling,
        scope="bounded_anomaly_detection",
    )


def run_budgeted_time_series(
    dataframe: pd.DataFrame,
    *,
    budget: ExecutionBudget,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Run time-series discovery/diagnostics on an order-preserving bounded frame."""
    sample_limit = max(30, min(budget.time_series_sample_rows, max(len(dataframe), 1)))
    work, sampling = deterministic_sample_frame(
        dataframe,
        sample_limit,
        preserve_order=True,
    )
    payload = detect_and_analyze_time_series(work, target_column=target_column)
    sampling = {
        **sampling,
        "reason": (
            "Time-series diagnostics use an order-preserving bounded view to avoid "
            "unbounded date parsing, stationarity, PACF, STL, and forecasting work."
            if sampling["sampled"]
            else "Full input fits within the time-series execution budget."
        ),
        "temporal_order_preserved": True,
    }
    return _attach_execution(
        payload,
        budget=budget,
        sampling=sampling,
        scope="bounded_time_series",
    )
