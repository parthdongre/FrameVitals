"""Full FrameVitals orchestration for streaming dataset sources.

The streaming pipeline performs one full-source Arrow scan to build reusable
profile state, retains one bounded deterministic working sample, and runs legacy
row-dependent modules only on that sample. Full-source facts are overlaid back
onto the stable analysis result shape, and every bounded module is explicitly
scoped so downstream consumers never mistake sample-derived diagnostics for
full-data execution.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from framevitals.analysis_selector import select_analyses
from framevitals.column_roles import summarize_roles
from framevitals.config import VALID_MODULES
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.execution import derive_execution_budget
from framevitals.health_score import calculate_health_score_from_profile_sample
from framevitals.ml_readiness import calculate_ml_readiness_from_profile
from framevitals.pipeline import run_full_analysis
from framevitals.signal_engine import build_signals
from framevitals.sources import StreamingDatasetSource
from framevitals.streaming_profile import build_streaming_profile
from framevitals.streaming_quality import run_streaming_quality_diagnostics
from framevitals.streaming_roles import infer_streaming_column_roles


_BOUNDED_RESULT_KEYS = (
    "advanced",
    "deep_statistics_v2",
    "anomalies_v2",
    "target_intelligence",
    "model_leaderboard",
    "explainability",
    "time_series",
    "text_profile",
)


def _working_sample_rows(budget) -> int:
    """Choose one reusable row sample large enough for every bounded module."""
    requested = max(
        budget.quality_sample_rows,
        budget.deep_statistics_sample_rows,
        budget.anomaly_sample_rows,
        budget.time_series_sample_rows,
        budget.pair_sample_rows,
        1,
    )
    return min(int(budget.rows), int(requested))


def _scope_bounded_result(
    value: Any,
    *,
    source_rows: int,
    sample_rows: int,
    strategy: str,
) -> Any:
    if not isinstance(value, dict):
        return value
    scoped = dict(value)
    scoped["execution_scope"] = {
        "scope": "bounded_row_sample" if sample_rows < source_rows else "full_source",
        "full_materialization": False,
        "source_rows": int(source_rows),
        "sample_rows": int(sample_rows),
        "sampled": bool(sample_rows < source_rows),
        "strategy": strategy,
    }
    return scoped


def _streaming_cleaning_result(
    *,
    enabled_by_user: bool,
    profile: dict[str, Any],
    health: dict[str, Any],
) -> dict[str, Any]:
    missing_count = sum(
        int(value)
        for value in profile.get("missing_counts", {}).values()
        if value is not None
    )
    duplicate_count = int(profile.get("duplicate_rows", 0) or 0)
    if enabled_by_user:
        reason = (
            "Full-source cleaning is not executed implicitly on the streaming analysis "
            "path because applying sample-derived mutations would be unsafe. Run the "
            "explicit cleaning workflow when a transformed dataset is required."
        )
        status = "deferred_streaming"
    else:
        reason = "Disabled by configuration."
        status = "disabled"
    return {
        "available": False,
        "skipped": True,
        "module": "cleaning",
        "reason": reason,
        "streaming_status": status,
        "actions": [],
        "before_health": health,
        "after_health": health,
        "output_path": None,
        "missing_before": missing_count,
        "missing_after": missing_count,
        "duplicates_before": duplicate_count,
        "duplicates_after": duplicate_count,
    }


def run_streaming_analysis(
    *,
    source: StreamingDatasetSource,
    dataset_id: str,
    original_filename: str,
    analysis_mode: str,
    target_column: str | None,
    parallel_workers: int,
    skip_ai: bool,
    disabled_modules: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run a stable-shape full analysis without materializing the complete source."""
    overall_start = time.perf_counter()
    metadata = source.inspect()
    rows = int(metadata.rows or 0)
    columns = int(metadata.columns or 0)
    if rows < 1 or columns < 1:
        raise ValueError(f"Dataset is empty or has no columns: {metadata.name}")

    user_disabled = set(disabled_modules or ())
    unknown = sorted(user_disabled - VALID_MODULES)
    if unknown:
        raise ValueError("Unknown disabled module(s): " + ", ".join(unknown))

    budget = derive_execution_budget(rows, columns, mode=analysis_mode)
    working_rows = _working_sample_rows(budget)

    profile_start = time.perf_counter()
    profile, working_sample = build_streaming_profile(
        source,
        sample_rows=working_rows,
        return_sample=True,
    )
    streaming_profile_ms = (time.perf_counter() - profile_start) * 1000

    if working_sample.empty:
        raise ValueError(f"Streaming source produced no usable rows: {metadata.name}")
    if target_column is not None and target_column not in working_sample.columns:
        raise ValueError(f"Target column not found: {target_column}")

    role_payload = infer_streaming_column_roles(working_sample, profile=profile)
    column_roles = role_payload["columns"]
    roles_summary = summarize_roles(column_roles)
    health = calculate_health_score_from_profile_sample(profile, working_sample)
    ml_readiness = calculate_ml_readiness_from_profile(profile)
    quality_diagnostics = (
        run_streaming_quality_diagnostics(
            working_sample,
            profile=profile,
            source_rows=rows,
            source_columns=columns,
            max_sample_rows=max(budget.quality_sample_rows, 10),
        )
        if "quality_diagnostics" not in user_disabled
        else {
            "available": False,
            "skipped": True,
            "module": "quality_diagnostics",
            "reason": "Disabled by configuration.",
        }
    )

    dataset_signals = detect_dataset_signals(
        working_sample,
        profile,
        column_roles=column_roles,
        source_shape=(rows, columns),
    )
    analysis_selection = select_analyses(
        signals=dataset_signals,
        analysis_mode=analysis_mode,
        target_column=target_column,
    )
    analysis_selection["execution_modules"] = {
        "disabled": sorted(user_disabled),
        "enabled": sorted(VALID_MODULES - user_disabled),
    }
    analysis_selection["execution_budget"] = budget.to_dict()
    analysis_selection["streaming"] = {
        "enabled": True,
        "full_materialization": False,
        "working_sample_rows": int(len(working_sample)),
        "source_rows": rows,
    }

    # Cleaning/charts are internally suppressed on the bounded sample. Charts
    # are irrelevant here because this path is only selected when artifacts are
    # disabled; cleaning would otherwise describe a transformed sample as though
    # it represented the complete source.
    internal_disabled = set(user_disabled) | {"cleaning", "charts"}
    sample_payload = run_full_analysis(
        dataset_id=dataset_id,
        original_filename=original_filename,
        analysis_mode=analysis_mode,
        skip_ai=skip_ai,
        target_column=target_column,
        parallel_workers=parallel_workers,
        dataframe=working_sample,
        write_artifacts=False,
        disabled_modules=tuple(sorted(internal_disabled)),
    )

    sample_rows = int(len(working_sample))
    sample_strategy = "streaming_evenly_spaced_global_rows"
    for key in _BOUNDED_RESULT_KEYS:
        sample_payload[key] = _scope_bounded_result(
            sample_payload.get(key),
            source_rows=rows,
            sample_rows=sample_rows,
            strategy=sample_strategy,
        )

    advanced = sample_payload.get("advanced")
    signals = build_signals(profile, health, ml_readiness, advanced)

    cleaning = _streaming_cleaning_result(
        enabled_by_user="cleaning" not in user_disabled,
        profile=profile,
        health=health,
    )

    execution = dict(sample_payload.get("execution", {}))
    module_status = dict(execution.get("module_status", {}))
    module_status["cleaning"] = (
        "disabled" if "cleaning" in user_disabled else "deferred_streaming"
    )
    module_status["charts"] = (
        "disabled" if "charts" in user_disabled else "not_applicable"
    )
    if "quality_diagnostics" in user_disabled:
        module_status["quality_diagnostics"] = "disabled"
    else:
        module_status["quality_diagnostics"] = "ran"

    module_scope: dict[str, str] = {}
    for module in (
        "deep_statistics",
        "anomaly_detection",
        "time_series",
        "text_profile",
        "target_intelligence",
        "modeling",
        "explainability",
    ):
        status = module_status.get(module)
        if status in {"ran", "scheduled", "error"}:
            module_scope[module] = (
                "bounded_row_sample" if sample_rows < rows else "full_source"
            )
    module_scope.update({
        "profile": "full_stream",
        "column_roles": "full_stream_plus_bounded_semantic_sample",
        "health": "full_stream_plus_bounded_outlier_sample",
        "ml_readiness": "full_stream_profile",
        "quality_diagnostics": (
            "disabled"
            if "quality_diagnostics" in user_disabled
            else "full_stream_plus_bounded_value_sample"
        ),
        "cleaning": "deferred_streaming" if "cleaning" not in user_disabled else "disabled",
        "charts": "not_applicable" if "charts" not in user_disabled else "disabled",
    })

    execution.update({
        "disabled_modules": sorted(user_disabled),
        "internally_disabled_modules": sorted(internal_disabled - user_disabled),
        "module_status": module_status,
        "module_scope": module_scope,
        "budget": budget.to_dict(),
        "streaming": {
            "enabled": True,
            "source": metadata.to_dict(),
            "full_materialization": False,
            "source_rows": rows,
            "source_columns": columns,
            "working_sample_rows": sample_rows,
            "working_sample_strategy": sample_strategy,
            "single_full_source_profile_scan": True,
        },
    })

    timings = dict(sample_payload.get("timings_ms", {}))
    timings["streaming_profile"] = streaming_profile_ms
    timings["total"] = (time.perf_counter() - overall_start) * 1000

    sample_payload.update({
        "filename": original_filename,
        "artifacts_enabled": False,
        "execution": execution,
        "profile": profile,
        "column_roles": column_roles,
        "roles_summary": roles_summary,
        "dataset_signals": dataset_signals,
        "analysis_selection": analysis_selection,
        "health": health,
        "signals": signals,
        "ml_readiness": ml_readiness,
        "quality_diagnostics": quality_diagnostics,
        "cleaning": cleaning,
        "charts": [],
        "timings_ms": timings,
    })
    return sample_payload
