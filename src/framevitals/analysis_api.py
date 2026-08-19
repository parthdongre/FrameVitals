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
from framevitals.pipeline import run_full_analysis
from framevitals.result import AnalysisResult
from framevitals.sources import StreamingDatasetSource, resolve_source


DataInput = Any


_MODE_DISABLED_MODULES: dict[str, frozenset[str]] = {
    # Quick is intentionally an overview. Keep explicit target intelligence and
    # artifact cleaning available for backwards compatibility, while omitting
    # the heavier anomaly/time-series/research/modeling layers.
    "quick": frozenset({
        "deep_statistics",
        "anomaly_detection",
        "time_series",
        "text_profile",
        "modeling",
        "explainability",
    }),
    # Standard is the operational default. It keeps practical anomaly,
    # time-series and target diagnostics, but leaves research-grade statistics,
    # free-text profiling and model training/explainability to deeper tiers.
    "standard": frozenset({
        "deep_statistics",
        "text_profile",
        "modeling",
        "explainability",
    }),
    # Deep is the advanced diagnostic tier: research-grade statistics and text
    # profiling are enabled, while repeated model CV/refitting is reserved for
    # research mode where the extra runtime is an explicit user choice.
    "deep": frozenset({"modeling", "explainability"}),
    "research": frozenset(),
}


def _effective_disabled_modules(
    mode: str,
    user_disabled: tuple[str, ...],
) -> tuple[str, ...]:
    """Merge explicit disables with the stable module policy for a mode."""
    implicit = _MODE_DISABLED_MODULES.get(mode)
    if implicit is None:
        raise ValueError(f"Unknown analysis mode: {mode}")
    return tuple(sorted(set(user_disabled) | set(implicit)))


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
    )
    effective_disabled = _effective_disabled_modules(
        resolved.mode,
        resolved.disabled_modules,
    )
    dataset_id = f"fv_{uuid4().hex[:12]}"

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
            from framevitals.streaming_exact_reuse import reuse_streaming_exact_statistics
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
    return AnalysisResult(payload)
