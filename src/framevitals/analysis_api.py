"""Public full-analysis dispatcher.

This module keeps input/source routing separate from the legacy analysis API.
Streaming-capable sources use the bounded streaming orchestrator when artifacts
are disabled; DataFrames and exact/materialized execution retain the existing
full pipeline behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from framevitals.config import ConfigInput, resolve_config
from framevitals.execution import ExecutionPolicy, use_execution_policy
from framevitals.pipeline import run_full_analysis
from framevitals.planner import MODE_DISABLED_MODULES, effective_disabled_modules
from framevitals.result import AnalysisResult
from framevitals.sources import StreamingDatasetSource, resolve_source


DataInput = Any

# Compatibility aliases retained for internal callers/tests that imported the
# pre-0.3 private policy names. The planner remains the single source of truth;
# these aliases deliberately do not duplicate policy data or behavior.
_MODE_DISABLED_MODULES = MODE_DISABLED_MODULES
_effective_disabled_modules = effective_disabled_modules


def analyze(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: ConfigInput = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
    max_sample_rows: int | None = None,
    max_relationship_pairs: int | None = None,
    max_memory_heavy_parallelism: int | None = None,
    max_streaming_profile_columns: int | None = None,
) -> AnalysisResult:
    """Analyze a tabular dataset through the appropriate execution source."""
    resolved = resolve_config(
        config,
        preset=preset,
        mode=mode,
        target=target,
        artifacts=artifacts,
        workers=workers,
        disabled_modules=disabled_modules,
        max_sample_rows=max_sample_rows,
        max_relationship_pairs=max_relationship_pairs,
        max_memory_heavy_parallelism=max_memory_heavy_parallelism,
        max_streaming_profile_columns=max_streaming_profile_columns,
    )
    effective_disabled = effective_disabled_modules(
        resolved.mode,
        resolved.disabled_modules,
    )
    dataset_id = f"fv_{uuid4().hex[:12]}"
    execution_policy = ExecutionPolicy(**resolved.execution_policy())

    # ExecutionPolicy is carried through a context variable so the existing
    # materialized and streaming pipelines derive the same capped budgets without
    # process-global mutation or a parallel configuration API.
    with use_execution_policy(execution_policy):
        # Preserve the direct DataFrame path so callers do not pay for a defensive
        # source-layer copy before the established materialized pipeline begins.
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("Dataset DataFrame is empty.")
            payload = run_full_analysis(
                dataset_id=dataset_id,
                original_filename="<dataframe>",
                analysis_mode=resolved.mode,
                target_column=resolved.target,
                parallel_workers=resolved.workers,
                skip_ai=True,
                dataframe=data,
                write_artifacts=resolved.artifacts,
                disabled_modules=effective_disabled,
            )
        else:
            source = resolve_source(data)
            metadata = source.inspect()
            if metadata.rows == 0:
                raise ValueError(f"Dataset is empty: {metadata.name}")

            if (
                metadata.supports_streaming
                and isinstance(source, StreamingDatasetSource)
                and not resolved.artifacts
            ):
                from framevitals.streaming_exact_reuse import (
                    reuse_streaming_exact_statistics,
                )
                from framevitals.streaming_pipeline import run_streaming_analysis

                payload = run_streaming_analysis(
                    source=source,
                    dataset_id=dataset_id,
                    original_filename=metadata.name,
                    analysis_mode=resolved.mode,
                    target_column=resolved.target,
                    parallel_workers=resolved.workers,
                    skip_ai=True,
                    disabled_modules=effective_disabled,
                )
                payload = reuse_streaming_exact_statistics(payload)
            elif isinstance(data, (str, Path)):
                path = Path(data)
                if not path.exists():
                    raise FileNotFoundError(f"Dataset not found: {path}")
                if not path.is_file():
                    raise ValueError(f"Expected a file for dataset, got: {path}")
                payload = run_full_analysis(
                    dataset_id=dataset_id,
                    file_path=path,
                    original_filename=metadata.name,
                    analysis_mode=resolved.mode,
                    target_column=resolved.target,
                    parallel_workers=resolved.workers,
                    skip_ai=True,
                    write_artifacts=resolved.artifacts,
                    disabled_modules=effective_disabled,
                )
            else:
                dataframe = source.load()
                payload = run_full_analysis(
                    dataset_id=dataset_id,
                    original_filename=metadata.name,
                    analysis_mode=resolved.mode,
                    target_column=resolved.target,
                    parallel_workers=resolved.workers,
                    skip_ai=True,
                    dataframe=dataframe,
                    write_artifacts=resolved.artifacts,
                    disabled_modules=effective_disabled,
                )

    # Keep public configuration/provenance compatible: ``disabled_modules``
    # describes only explicit caller configuration. Mode policy is observable
    # through ``execution.module_status`` and analysis selection instead of
    # masquerading as a user-supplied disable list.
    payload["config"] = resolved.to_dict()
    execution = payload.get("execution")
    if isinstance(execution, dict):
        execution["disabled_modules"] = sorted(resolved.disabled_modules)
        execution["resource_policy"] = execution_policy.to_dict()
        execution["effective_disabled_modules"] = list(effective_disabled)
    return AnalysisResult(payload)
