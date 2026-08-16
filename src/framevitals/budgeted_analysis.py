"""Resource-bounded adapters around expensive analysis modules.

These adapters make legacy analyses safe on large inputs while FrameVitals moves
more work into streaming/native kernels. Sampling and adaptive column selection
are deterministic and are always disclosed in returned execution metadata.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from framevitals.anomaly_ensemble import detect_anomalies_ensemble
from framevitals.deep_statistics_v2 import run_deep_statistics_v2
from framevitals.deep_triage import triage_deep_columns
from framevitals.execution import ExecutionBudget, deterministic_sample_frame
from framevitals.fast_anomaly import fast_anomaly_scan
from framevitals.neural_anomaly import neural_reconstruction_anomalies
from framevitals.provenance import normalize_execution
from framevitals.stream_change import scan_ordered_mean_shift
from framevitals.time_series import detect_and_analyze_time_series


def _attach_execution(
    payload: dict[str, Any],
    *,
    budget: ExecutionBudget,
    sampling: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    result = dict(payload)
    result["execution"] = normalize_execution(
        {
            "scope": scope,
            "scale_class": budget.scale_class,
            **sampling,
        },
        method=scope,
        full_materialization=False,
    )
    return result


def run_budgeted_deep_statistics(
    dataframe: pd.DataFrame,
    *,
    budget: ExecutionBudget,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Run deep statistics on a bounded, adaptively selected diagnostic view."""
    sample_limit = max(
        20,
        min(
            budget.deep_statistics_sample_rows,
            budget.bootstrap_sample_rows,
        ),
    )
    sample_limit = min(sample_limit, max(len(dataframe), 1))
    work, sampling = deterministic_sample_frame(dataframe, sample_limit)

    triage = triage_deep_columns(work, mode=budget.mode)
    selected_columns = list(triage.selected_columns)
    diagnostic_view = (
        work.loc[:, selected_columns]
        if selected_columns
        else pd.DataFrame(index=work.index)
    )

    pair_budget = (
        budget.relationship_pair_budget
        if max_pairs is None
        else min(int(max_pairs), budget.relationship_pair_budget)
    )
    if pair_budget < 1:
        raise ValueError("max_pairs must be at least 1.")

    payload = run_deep_statistics_v2(diagnostic_view, max_pairs=pair_budget)
    triage_payload = triage.to_dict()
    payload["column_triage"] = triage_payload

    sampling = {
        **sampling,
        "reason": (
            "Deep statistics use a bounded row view plus adaptive column triage; "
            "bootstrap/distribution work is reserved for the highest-interest columns."
        ),
        "pair_budget": int(pair_budget),
        "column_triage": triage_payload,
        "source_columns": int(dataframe.shape[1]),
        "diagnostic_columns": int(len(selected_columns)),
        "adaptive_strategy": "column_interest_triage",
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
    """Run fast screening in standard/deep and full confirmation in research."""
    sample_limit = max(20, min(budget.anomaly_sample_rows, max(len(dataframe), 1)))
    work, sampling = deterministic_sample_frame(dataframe, sample_limit)

    if budget.mode == "research":
        payload = detect_anomalies_ensemble(
            work,
            contamination=contamination,
            threshold=threshold,
            max_columns=max_columns,
            top_k=top_k,
        )
        anomaly_strategy = "classical_ensemble_plus_neural_reconstruction"
        try:
            payload["neural_reconstruction"] = neural_reconstruction_anomalies(
                work,
                max_rows=min(3_000, max(len(work), 20)),
                max_columns=min(max_columns, 24),
                max_iter=35,
                top_k=top_k,
            )
        except Exception as exc:  # neural diagnostics must fail soft
            payload["neural_reconstruction"] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
    else:
        payload = fast_anomaly_scan(
            work,
            contamination=contamination,
            threshold=threshold,
            max_columns=min(max_columns, 24),
            projections=12,
            top_k=top_k,
        )
        anomaly_strategy = "fast_robust_random_projection"

    sampling = {
        **sampling,
        "reason": (
            "Research mode confirms anomalies with the heavier classical ensemble and "
            "a bounded neural reconstruction detector."
            if budget.mode == "research"
            else "Standard/deep modes use vectorized robust and random-projection anomaly screening."
        ),
        "coverage": "sample" if sampling["sampled"] else "full",
        "anomaly_strategy": anomaly_strategy,
        "adaptive_strategy": anomaly_strategy,
        "neural_reconstruction_enabled": budget.mode == "research",
    }
    return _attach_execution(
        payload,
        budget=budget,
        sampling=sampling,
        scope="bounded_anomaly_detection",
    )


def _attach_time_series_change_scan(
    payload: dict[str, Any],
    work: pd.DataFrame,
) -> bool:
    """Attach a cheap ordered mean-shift scan when time-series detection succeeded."""
    if not payload.get("available"):
        return False
    date_column = payload.get("detected_date_column")
    numeric_column = payload.get("numeric_column")
    if not isinstance(date_column, str) or not isinstance(numeric_column, str):
        return False
    if date_column not in work.columns or numeric_column not in work.columns:
        return False

    parsed_dates = pd.to_datetime(work[date_column], errors="coerce", format="mixed")
    ordered = pd.DataFrame({
        "date": parsed_dates,
        "value": pd.to_numeric(work[numeric_column], errors="coerce"),
    }).dropna()
    if ordered.empty:
        return False
    ordered = ordered.sort_values("date")
    payload["mean_shift"] = scan_ordered_mean_shift(
        ordered["value"],
        windows=24,
        threshold=8.0,
        min_updates=8,
    )
    return True


def run_budgeted_time_series(
    dataframe: pd.DataFrame,
    *,
    budget: ExecutionBudget,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Run bounded ordered time-series diagnostics plus change detection."""
    sample_limit = max(30, min(budget.time_series_sample_rows, max(len(dataframe), 1)))
    work, sampling = deterministic_sample_frame(
        dataframe,
        sample_limit,
        preserve_order=True,
    )
    payload = detect_and_analyze_time_series(work, target_column=target_column)
    change_detection_enabled = _attach_time_series_change_scan(payload, work)
    sampling = {
        **sampling,
        "reason": (
            "Time-series diagnostics use an order-preserving bounded view to avoid "
            "unbounded date parsing, stationarity, PACF, STL, and forecasting work. "
            "Detected numeric series also receive a bounded Page-Hinkley mean-shift scan."
            if sampling["sampled"]
            else (
                "Full input fits within the time-series execution budget; detected numeric "
                "series also receive a bounded Page-Hinkley mean-shift scan."
            )
        ),
        "temporal_order_preserved": True,
        "mean_shift_detection_enabled": bool(change_detection_enabled),
        "adaptive_strategy": "ordered_page_hinkley_mean_shift",
    }
    return _attach_execution(
        payload,
        budget=budget,
        sampling=sampling,
        scope="bounded_time_series",
    )
