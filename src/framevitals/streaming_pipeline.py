"""Full FrameVitals orchestration for streaming dataset sources.

The streaming pipeline performs one full-source Arrow scan to build reusable
profile state, retains one bounded deterministic working sample, and runs only
row-dependent modules on that sample. Full-source facts are reused instead of
re-entering the materialized pandas pipeline, and every bounded module is
explicitly scoped so downstream consumers never mistake sample-derived
diagnostics for full-data execution.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

from framevitals.analysis_selector import select_analyses
from framevitals.column_roles import summarize_roles
from framevitals.config import VALID_MODULES
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.execution import (
    derive_execution_budget,
    derive_streaming_profile_column_limit,
)
from framevitals.health_score import calculate_health_score_from_profile_sample
from framevitals.ml_readiness import calculate_ml_readiness_from_profile
from framevitals.signal_engine import build_signals
from framevitals.sources import DatasetMetadata, StreamingDatasetSource
from framevitals.streaming_bounded_pipeline import run_streaming_bounded_modules
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


def _evenly_spaced_names(names: Sequence[str], limit: int) -> list[str]:
    total = len(names)
    if total <= limit:
        return list(names)
    if limit <= 1:
        return [str(names[0])]
    selected = [
        str(names[(index * (total - 1)) // (limit - 1)])
        for index in range(limit)
    ]
    return list(dict.fromkeys(selected))


def _streaming_profile_projection(
    source: StreamingDatasetSource,
    *,
    source_columns: int,
    limit: int,
    target_column: str | None,
) -> list[str] | None:
    """Choose a deterministic schema-wide projection for ultra-wide sources."""
    if source_columns <= limit:
        return None

    schema_method = getattr(source, "schema", None)
    if not callable(schema_method):
        raise TypeError(
            "Ultra-wide streaming analysis requires source.schema() so FrameVitals "
            "can project columns instead of scanning the complete width."
        )
    schema = schema_method()
    names = [str(field.name) for field in schema]
    if target_column is not None and target_column not in names:
        raise ValueError(f"Target column not found: {target_column}")

    selected = _evenly_spaced_names(names, limit)
    if target_column is not None and target_column not in selected:
        selected[-1] = target_column
    return list(dict.fromkeys(selected))


class _ProjectedStreamingSource:
    """Projection-enforcing view over a streaming source.

    The wrapped source is never asked for the complete ultra-wide width. This
    keeps the reusable full-row profile scan bounded while preserving source
    metadata in the outer pipeline.
    """

    def __init__(self, source: StreamingDatasetSource, columns: Sequence[str]):
        self._source = source
        self._columns = tuple(columns)

    def inspect(self) -> DatasetMetadata:
        metadata = self._source.inspect()
        return DatasetMetadata(
            name=metadata.name,
            kind=metadata.kind,
            format=metadata.format,
            rows=metadata.rows,
            columns=len(self._columns),
            size_bytes=metadata.size_bytes,
            materialized=metadata.materialized,
            supports_projection=True,
            supports_streaming=True,
        )

    def schema(self):
        schema_method = getattr(self._source, "schema", None)
        if not callable(schema_method):
            raise TypeError("Projected streaming sources require schema().")
        schema = schema_method()
        return [schema.field(name) for name in self._columns]

    def iter_batches(
        self,
        *,
        batch_size: int = 65_536,
        columns: Sequence[str] | None = None,
    ):
        requested = list(self._columns if columns is None else columns)
        allowed = set(self._columns)
        unknown = [column for column in requested if column not in allowed]
        if unknown:
            raise ValueError(
                "Projected source requested columns outside its budget: "
                + ", ".join(unknown[:5])
            )
        yield from self._source.iter_batches(
            batch_size=batch_size,
            columns=requested,
        )

    def load(self):
        raise RuntimeError("Projected streaming source must never materialize fully.")


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
    profile_column_limit = derive_streaming_profile_column_limit(
        rows,
        columns,
        mode=analysis_mode,
    )
    profile_columns = _streaming_profile_projection(
        source,
        source_columns=columns,
        limit=profile_column_limit,
        target_column=target_column,
    )
    profile_source: StreamingDatasetSource = (
        _ProjectedStreamingSource(source, profile_columns)
        if profile_columns is not None
        else source
    )

    core_timings: dict[str, float] = {}
    t0 = time.perf_counter()
    profile, working_sample = build_streaming_profile(
        profile_source,
        sample_rows=working_rows,
        return_sample=True,
    )
    core_timings["streaming_profile"] = (time.perf_counter() - t0) * 1000

    profiled_columns = int(len(working_sample.columns))
    column_sampled = profiled_columns < columns
    profile["shape"] = {"rows": rows, "columns": columns}
    profile["source_metadata"] = metadata.to_dict()
    streaming_metadata = dict(profile.get("streaming_metadata", {}))
    streaming_metadata.update({
        "source_columns": columns,
        "profiled_columns": profiled_columns,
        "column_sampled": column_sampled,
        "column_limit": int(profile_column_limit),
        "column_strategy": (
            "deterministic_schema_projection" if column_sampled else "full_schema"
        ),
    })
    profile["streaming_metadata"] = streaming_metadata

    if working_sample.empty:
        raise ValueError(f"Streaming source produced no usable rows: {metadata.name}")
    if target_column is not None and target_column not in working_sample.columns:
        raise ValueError(f"Target column not found: {target_column}")

    t0 = time.perf_counter()
    role_payload = infer_streaming_column_roles(working_sample, profile=profile)
    column_roles = role_payload["columns"]
    roles_summary = summarize_roles(column_roles)
    core_timings["column_roles"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    health = calculate_health_score_from_profile_sample(profile, working_sample)
    core_timings["health"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ml_readiness = calculate_ml_readiness_from_profile(profile)
    core_timings["ml_readiness"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
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
    core_timings["quality_diagnostics"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dataset_signals = detect_dataset_signals(
        working_sample,
        profile,
        column_roles=column_roles,
        source_shape=(rows, columns),
    )
    core_timings["dataset_signals"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
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
        "source_columns": columns,
        "profiled_columns": profiled_columns,
        "column_sampled": column_sampled,
    }
    core_timings["analysis_selection"] = (time.perf_counter() - t0) * 1000

    # Cleaning/charts are internally suppressed on the bounded sample. The
    # streaming scheduler runs only modules that genuinely require row values;
    # profile/roles/health/readiness/quality above are never recomputed.
    internal_disabled = set(user_disabled) | {"cleaning", "charts"}
    sample_payload = run_streaming_bounded_modules(
        working_sample,
        dataset_id=dataset_id,
        original_filename=original_filename,
        analysis_mode=analysis_mode,
        target_column=target_column,
        parallel_workers=parallel_workers,
        source_budget=budget,
        column_roles=column_roles,
        skip_ai=skip_ai,
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
    t0 = time.perf_counter()
    signals = build_signals(profile, health, ml_readiness, advanced)
    core_timings["signals"] = (time.perf_counter() - t0) * 1000

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
    module_status["quality_diagnostics"] = (
        "disabled" if "quality_diagnostics" in user_disabled else "ran"
    )

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
    profile_scope = (
        "full_rows_projected_columns" if column_sampled else "full_stream"
    )
    module_scope.update({
        "profile": profile_scope,
        "column_roles": (
            "full_rows_projected_columns_plus_bounded_semantic_sample"
            if column_sampled
            else "full_stream_plus_bounded_semantic_sample"
        ),
        "health": (
            "full_rows_projected_columns_plus_bounded_outlier_sample"
            if column_sampled
            else "full_stream_plus_bounded_outlier_sample"
        ),
        "ml_readiness": (
            "full_rows_projected_columns_profile"
            if column_sampled
            else "full_stream_profile"
        ),
        "quality_diagnostics": (
            "disabled"
            if "quality_diagnostics" in user_disabled
            else (
                "full_rows_projected_columns_plus_bounded_value_sample"
                if column_sampled
                else "full_stream_plus_bounded_value_sample"
            )
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
            "profiled_columns": profiled_columns,
            "column_sampled": column_sampled,
            "column_limit": int(profile_column_limit),
            "column_strategy": (
                "deterministic_schema_projection" if column_sampled else "full_schema"
            ),
            "working_sample_rows": sample_rows,
            "working_sample_strategy": sample_strategy,
            "single_full_source_profile_scan": True,
        },
    })

    timings = dict(sample_payload.get("timings_ms", {}))
    timings.update(core_timings)
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
