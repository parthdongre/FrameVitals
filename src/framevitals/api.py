from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from framevitals.config import ConfigInput, resolve_config
from framevitals.contracts import infer_contract as _infer_contract
from framevitals.contracts import validate_contract
from framevitals.drift_analysis import compare_datasets
from framevitals.loader import load_dataset
from framevitals.pipeline import run_full_analysis
from framevitals.result import AnalysisResult

DataInput = str | Path | pd.DataFrame


def _validated_path(value: str | Path, *, label: str = "Dataset") -> Path:
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file for {label.lower()}, got: {path}")
    return path


def _load_input(value: DataInput, *, label: str) -> tuple[pd.DataFrame, str]:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            raise ValueError(f"{label} DataFrame is empty.")
        return value.copy(), "<dataframe>"

    if isinstance(value, (str, Path)):
        path = _validated_path(value, label=label)
        df = load_dataset(path)
        if df.empty:
            raise ValueError(f"{label} dataset is empty: {path}")
        return df, path.name

    raise TypeError(
        f"{label} must be a pandas DataFrame or a path to a supported dataset."
    )


def analyze(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: ConfigInput = None,
) -> AnalysisResult:
    """Analyze a tabular dataset with FrameVitals.

    Parameters
    ----------
    data:
        A pandas DataFrame or a path to a CSV, TSV, Excel, or JSON dataset.
    target:
        Optional supervised-learning target column. Explicit values override
        configuration-file values.
    mode:
        Analysis depth: ``quick``, ``standard``, ``deep``, or ``research``.
        ``None`` resolves to configuration/preset/default behaviour.
    artifacts:
        Persist cleaned CSV/chart artifacts. ``None`` resolves from config and
        otherwise defaults to ``False`` for reusable library calls.
    workers:
        Worker count for independent diagnostic tasks.
    preset:
        Optional built-in runtime preset such as ``quick``, ``standard``,
        ``deep``, ``research``, or ``ci``.
    config:
        ``AnalysisConfig``, mapping, or path to a TOML configuration file.

    Returns
    -------
    AnalysisResult
        A dict-compatible structured result with summary, column, JSON, HTML,
        and notebook helpers.
    """
    resolved = resolve_config(
        config,
        preset=preset,
        mode=mode,
        target=target,
        artifacts=artifacts,
        workers=workers,
    )
    dataset_id = f"fv_{uuid4().hex[:12]}"

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Dataset DataFrame is empty.")
        payload = run_full_analysis(
            dataset_id=dataset_id,
            dataframe=data,
            original_filename="<dataframe>",
            analysis_mode=resolved.mode,
            target_column=resolved.target,
            parallel_workers=resolved.workers,
            skip_ai=True,
            write_artifacts=resolved.artifacts,
        )
    elif isinstance(data, (str, Path)):
        path = _validated_path(data)
        payload = run_full_analysis(
            dataset_id=dataset_id,
            file_path=path,
            original_filename=path.name,
            analysis_mode=resolved.mode,
            target_column=resolved.target,
            parallel_workers=resolved.workers,
            skip_ai=True,
            write_artifacts=resolved.artifacts,
        )
    else:
        raise TypeError(
            "data must be a pandas DataFrame or a path to a supported dataset."
        )

    payload["config"] = resolved.to_dict()
    return AnalysisResult(payload)


def compare(
    reference: DataInput,
    current: DataInput,
    *,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> dict:
    """Compare reference and current datasets for distribution drift.

    Both inputs may independently be pandas DataFrames or supported dataset
    paths. Numeric features use PSI, KS statistics, and standardized mean
    shift; categorical features use PSI and chi-square diagnostics.
    """
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    ref_df, ref_name = _load_input(reference, label="Reference")
    cur_df, cur_name = _load_input(current, label="Current")

    result = compare_datasets(
        ref_df,
        cur_df,
        columns=columns,
        max_columns=max_columns,
    )
    result["reference_name"] = ref_name
    result["current_name"] = cur_name
    return result


def infer_contract(data: DataInput) -> dict[str, Any]:
    """Infer a JSON-serializable data contract from a reference dataset.

    The result can be saved with :mod:`json` and passed to :func:`validate`
    when checking later datasets in a pipeline or CI job.
    """
    dataframe, source_name = _load_input(data, label="Reference")
    contract = _infer_contract(dataframe)
    contract["reference_name"] = source_name
    return contract


def validate(
    data: DataInput,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a dataset against an inferred or explicit data contract.

    Contract failures are returned as structured findings rather than raised,
    allowing callers to decide whether warnings or errors should block a job.
    Invalid contract definitions and unreadable datasets still raise clear
    exceptions.
    """
    dataframe, source_name = _load_input(data, label="Dataset")
    result = validate_contract(dataframe, contract)
    result["dataset_name"] = source_name
    return result
