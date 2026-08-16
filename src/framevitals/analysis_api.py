"""Public full-analysis dispatcher.

This module keeps input/source routing separate from the legacy analysis API.
Streaming Arrow-capable sources use the bounded streaming orchestrator when
artifacts are disabled; DataFrames, non-streaming files, and explicit artifact
runs retain the existing materialized pipeline behavior.
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


DataInput = str | Path | pd.DataFrame


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
    dataset_id = f"fv_{uuid4().hex[:12]}"

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
            disabled_modules=resolved.disabled_modules,
        )
    elif isinstance(data, (str, Path)):
        source = resolve_source(data)
        metadata = source.inspect()
        if metadata.rows == 0:
            raise ValueError(f"Dataset is empty: {metadata.name}")

        if (
            metadata.supports_streaming
            and isinstance(source, StreamingDatasetSource)
            and not resolved.artifacts
        ):
            from framevitals.streaming_pipeline import run_streaming_analysis

            payload = run_streaming_analysis(
                source=source,
                dataset_id=dataset_id,
                original_filename=metadata.name,
                analysis_mode=resolved.mode,
                target_column=resolved.target,
                parallel_workers=resolved.workers,
                skip_ai=True,
                disabled_modules=resolved.disabled_modules,
            )
        else:
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path}")
            if not path.is_file():
                raise ValueError(f"Expected a file for dataset, got: {path}")
            payload = run_full_analysis(
                dataset_id=dataset_id,
                file_path=path,
                original_filename=path.name,
                analysis_mode=resolved.mode,
                target_column=resolved.target,
                parallel_workers=resolved.workers,
                skip_ai=True,
                write_artifacts=resolved.artifacts,
                disabled_modules=resolved.disabled_modules,
            )
    else:
        raise TypeError(
            "data must be a pandas DataFrame or a path to a supported dataset."
        )

    payload["config"] = resolved.to_dict()
    return AnalysisResult(payload)
