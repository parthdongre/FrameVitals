"""Planning-only public execution path.

`fv.plan()` should be cheap enough to call before committing to a full analysis.
This module deliberately avoids importing the full FrameVitals pipeline while
still resolving configuration, structural signals, and adaptive execution
budgets exactly as the execution layer expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from framevitals.column_roles import infer_column_roles
from framevitals.config import AnalysisConfig, ConfigInput, resolve_config
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.execution import (
    ExecutionPolicy,
    derive_execution_budget,
    derive_streaming_profile_column_limit,
    use_execution_policy,
)
from framevitals.planner import build_execution_plan
from framevitals.planning import AnalysisPlan
from framevitals.profiler import build_profile
from framevitals.sources import StreamingDatasetSource, resolve_source


DataInput = str | Path | pd.DataFrame
PLANNING_SAMPLE_ROWS = 5_000


def _evenly_spaced_names(names: Sequence[str], limit: int) -> list[str]:
    """Select a deterministic schema-wide column projection without NumPy."""
    total = len(names)
    if limit < 1:
        raise ValueError("column projection limit must be at least 1")
    if total <= limit:
        return list(names)
    if limit == 1:
        return [str(names[0])]

    selected = [
        str(names[(index * (total - 1)) // (limit - 1)])
        for index in range(limit)
    ]
    return list(dict.fromkeys(selected))


def _streaming_projection(
    source: StreamingDatasetSource,
    *,
    source_columns: int,
    limit: int,
    target: str | None,
) -> list[str] | None:
    """Resolve a bounded projection for ultra-wide streaming planning."""
    if source_columns <= limit:
        return None

    schema_method = getattr(source, "schema", None)
    if not callable(schema_method):
        raise TypeError(
            "Ultra-wide streaming planning requires source.schema() so FrameVitals "
            "can project columns instead of materializing the complete width."
        )
    schema = schema_method()
    names = [str(field.name) for field in schema]
    if target is not None and target not in names:
        raise ValueError(f"Target column not found: {target}")

    selected = _evenly_spaced_names(names, limit)
    if target is not None and target not in selected:
        if selected:
            selected[-1] = target
        else:
            selected = [target]
    return list(dict.fromkeys(selected))


def _streaming_head_sample(
    source: StreamingDatasetSource,
    *,
    max_rows: int = PLANNING_SAMPLE_ROWS,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read at most ``max_rows`` rows and stop; planning must not scan the file."""
    frames: list[pd.DataFrame] = []
    remaining = int(max_rows)
    for batch in source.iter_batches(batch_size=max_rows, columns=columns):
        if remaining <= 0:
            break
        take = min(remaining, int(batch.num_rows))
        if take > 0:
            frames.append(batch.slice(0, take).to_pandas())
            remaining -= take
        if remaining <= 0:
            break
    if not frames:
        raise ValueError("Streaming dataset produced no rows for planning.")
    return pd.concat(frames, ignore_index=True)


def _project_sample_profile_to_source(
    profile: dict[str, Any],
    *,
    source_rows: int,
    source_columns: int,
    sample_rows: int,
    profiled_columns: int,
) -> dict[str, Any]:
    """Scale rate-based sample metrics to the true source shape for planning."""
    projected = dict(profile)
    factor = source_rows / max(sample_rows, 1)
    projected["shape"] = {"rows": int(source_rows), "columns": int(source_columns)}
    projected["missing_counts"] = {
        column: int(round(float(value or 0) * factor))
        for column, value in profile.get("missing_counts", {}).items()
    }
    sample_duplicate_rows = int(profile.get("duplicate_rows", 0) or 0)
    projected["duplicate_rows"] = int(round(sample_duplicate_rows * factor))
    projected["planning_sample_metadata"] = {
        "sampled": sample_rows < source_rows,
        "sample_rows": int(sample_rows),
        "source_rows": int(source_rows),
        "strategy": "bounded_head",
        "full_scan": False,
        "rate_metrics_projected": sample_rows < source_rows,
        "profiled_columns": int(profiled_columns),
        "source_columns": int(source_columns),
        "column_sampled": int(profiled_columns) < int(source_columns),
    }
    return projected


def _build_plan(
    data: DataInput,
    *,
    resolved: AnalysisConfig,
    execution_policy: ExecutionPolicy,
) -> AnalysisPlan:
    source = resolve_source(data)
    source_metadata = source.inspect()

    if source_metadata.supports_streaming and isinstance(
        source, StreamingDatasetSource
    ):
        source_rows = int(source_metadata.rows or 0)
        source_columns = int(source_metadata.columns or 0)
        if source_rows < 1 or source_columns < 1:
            raise ValueError(
                f"Dataset is empty or has no columns: {source_metadata.name}"
            )

        column_limit = derive_streaming_profile_column_limit(
            source_rows,
            source_columns,
            mode=resolved.mode,
        )
        projected_columns = _streaming_projection(
            source,
            source_columns=source_columns,
            limit=column_limit,
            target=resolved.target,
        )
        planning_sample_rows = PLANNING_SAMPLE_ROWS
        if execution_policy.max_sample_rows is not None:
            planning_sample_rows = min(
                planning_sample_rows,
                int(execution_policy.max_sample_rows),
            )
        dataframe = _streaming_head_sample(
            source,
            max_rows=planning_sample_rows,
            columns=projected_columns,
        )
        profiled_columns = int(len(dataframe.columns))
        dataset_profile = _project_sample_profile_to_source(
            build_profile(dataframe),
            source_rows=source_rows,
            source_columns=source_columns,
            sample_rows=len(dataframe),
            profiled_columns=profiled_columns,
        )
        planning_data = {
            "materialized_full_dataset": False,
            "sampled": len(dataframe) < source_rows,
            "sample_rows": int(len(dataframe)),
            "source_rows": source_rows,
            "strategy": "bounded_head",
            "full_scan": False,
            "source_columns": source_columns,
            "sample_columns": profiled_columns,
            "column_sampled": profiled_columns < source_columns,
            "column_strategy": (
                "deterministic_schema_projection"
                if profiled_columns < source_columns
                else "full_schema"
            ),
            "column_limit": int(column_limit),
        }
    else:
        dataframe = source.load()
        source_rows = int(len(dataframe))
        source_columns = int(len(dataframe.columns))
        dataset_profile = build_profile(dataframe)
        planning_data = {
            "materialized_full_dataset": True,
            "sampled": False,
            "sample_rows": source_rows,
            "source_rows": source_rows,
            "strategy": "full_dataset",
            "full_scan": True,
            "source_columns": source_columns,
            "sample_columns": source_columns,
            "column_sampled": False,
            "column_strategy": "full_schema",
            "column_limit": source_columns,
        }

    source_columns_list = list(dataset_profile.get("columns", dataframe.columns))
    if resolved.target is not None and resolved.target not in source_columns_list:
        raise ValueError(f"Target column not found: {resolved.target}")

    column_roles = infer_column_roles(dataframe)
    dataset_signals = detect_dataset_signals(
        dataframe,
        dataset_profile,
        column_roles=column_roles,
        source_shape=(source_rows, source_columns),
    )

    budget = derive_execution_budget(
        source_rows,
        source_columns,
        mode=resolved.mode,
    )

    selection = build_execution_plan(
        signals=dataset_signals,
        analysis_mode=resolved.mode,
        target_column=resolved.target,
        disabled_modules=resolved.disabled_modules,
        artifacts=resolved.artifacts,
    )
    selection["execution_budget"] = budget.to_dict()

    public_signals = {
        key: value
        for key, value in dataset_signals.items()
        if key != "column_roles"
    }

    return AnalysisPlan({
        "dataset_name": source_metadata.name,
        "source": source_metadata.to_dict(),
        "planning_data": planning_data,
        "analysis_mode": resolved.mode,
        "target": resolved.target,
        "shape": dict(dataset_profile.get("shape", {})),
        "config": resolved.to_dict(),
        "resource_policy": execution_policy.to_dict(),
        "execution_budget": budget.to_dict(),
        "signals": public_signals,
        "selection": selection,
    })


def plan(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: ConfigInput = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
) -> AnalysisPlan:
    """Preview planned analyses, scale policy, and execution constraints."""
    resolved = resolve_config(
        config,
        preset=preset,
        mode=mode,
        target=target,
        workers=workers,
        artifacts=False,
        disabled_modules=disabled_modules,
    )
    execution_policy = ExecutionPolicy(**resolved.execution_policy())
    with use_execution_policy(execution_policy):
        return _build_plan(
            data,
            resolved=resolved,
            execution_policy=execution_policy,
        )
