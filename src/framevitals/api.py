from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from framevitals.analysis_selector import select_analyses
from framevitals.cleaning_plan import (
    CleaningPlan,
    apply_cleaning_plan,
    infer_cleaning_plan,
)
from framevitals.column_roles import infer_column_roles
from framevitals.config import ConfigInput, resolve_config
from framevitals.contracts import infer_contract as _infer_contract
from framevitals.contracts import validate_contract
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.drift_analysis import compare_datasets
from framevitals.loader import load_dataset
from framevitals.pipeline import run_full_analysis
from framevitals.planning import AnalysisPlan
from framevitals.profiler import build_profile
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
    """Analyze a tabular dataset with FrameVitals."""
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


def plan(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: ConfigInput = None,
) -> AnalysisPlan:
    """Preview which analyses FrameVitals considers applicable.

    Planning performs loading, profiling, role inference, signal detection, and
    selector evaluation only. It does not run the heavier model/statistics,
    cleaning, chart, or AI stages of :func:`analyze`.
    """
    resolved = resolve_config(
        config,
        preset=preset,
        mode=mode,
        target=target,
        workers=workers,
        artifacts=False,
    )
    dataframe, source_name = _load_input(data, label="Dataset")
    profile = build_profile(dataframe)
    column_roles = infer_column_roles(dataframe)
    dataset_signals = detect_dataset_signals(
        dataframe,
        profile,
        column_roles=column_roles,
    )
    selection = select_analyses(
        signals=dataset_signals,
        analysis_mode=resolved.mode,
        target_column=resolved.target,
    )

    public_signals = {
        key: value
        for key, value in dataset_signals.items()
        if key != "column_roles"
    }
    return AnalysisPlan({
        "dataset_name": source_name,
        "analysis_mode": resolved.mode,
        "target": resolved.target,
        "shape": dict(profile.get("shape", {})),
        "config": resolved.to_dict(),
        "signals": public_signals,
        "selection": selection,
    })


def plan_cleaning(data: DataInput) -> CleaningPlan:
    """Infer a conservative cleaning plan without modifying the input data."""
    dataframe, _ = _load_input(data, label="Dataset")
    profile = build_profile(dataframe)
    return infer_cleaning_plan(dataframe, profile=profile)


def clean(
    data: DataInput,
    *,
    plan: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return an explicitly cleaned copy of a dataset.

    If ``plan`` is omitted FrameVitals infers its conservative default plan.
    The original DataFrame or source file is never modified in place.
    """
    dataframe, _ = _load_input(data, label="Dataset")
    resolved_plan = plan if plan is not None else infer_cleaning_plan(dataframe)
    return apply_cleaning_plan(dataframe, resolved_plan, copy=True)


def compare(
    reference: DataInput,
    current: DataInput,
    *,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> dict:
    """Compare reference and current datasets for distribution drift."""
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
    """Infer a JSON-serializable data contract from a reference dataset."""
    dataframe, source_name = _load_input(data, label="Reference")
    contract = _infer_contract(dataframe)
    contract["reference_name"] = source_name
    return contract


def validate(
    data: DataInput,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a dataset against an inferred or explicit data contract."""
    dataframe, source_name = _load_input(data, label="Dataset")
    result = validate_contract(dataframe, contract)
    result["dataset_name"] = source_name
    return result
