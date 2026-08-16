from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from framevitals.analysis_selector import select_analyses
from framevitals.anomaly_ensemble import detect_anomalies_ensemble
from framevitals.cleaning_plan import (
    CleaningPlan,
    apply_cleaning_plan,
    infer_cleaning_plan,
)
from framevitals.column_roles import infer_column_roles, summarize_roles
from framevitals.config import ConfigInput, VALID_MODULES, resolve_config
from framevitals.contracts import infer_contract as _infer_contract
from framevitals.contracts import validate_contract
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.deep_statistics_v2 import run_deep_statistics_v2
from framevitals.drift_analysis import compare_datasets, severity_at_least
from framevitals.health_score import calculate_health_score
from framevitals.loader import load_dataset
from framevitals.ml_readiness import calculate_ml_readiness
from framevitals.pipeline import run_full_analysis
from framevitals.planning import AnalysisPlan
from framevitals.profiler import build_profile
from framevitals.quality_diagnostics import run_quality_diagnostics
from framevitals.quality_results import DriftResult, GateResult, ValidationResult
from framevitals.result import AnalysisResult
from framevitals.target_intelligence import run_target_intelligence

DataInput = str | Path | pd.DataFrame
_DRIFT_SEVERITY_RANK = {"stable": 0, "minor": 1, "moderate": 2, "severe": 3}


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


def _with_dataset_name(payload: dict[str, Any], source_name: str) -> dict[str, Any]:
    """Attach source identity without mutating an analysis module's own result."""
    return {"dataset_name": source_name, **payload}


def profile(data: DataInput) -> dict[str, Any]:
    """Profile shape, types, missingness, cardinality, and basic summaries only."""
    dataframe, source_name = _load_input(data, label="Dataset")
    return _with_dataset_name(build_profile(dataframe), source_name)


def roles(data: DataInput) -> dict[str, Any]:
    """Infer semantic/structural column roles without running the full pipeline."""
    dataframe, source_name = _load_input(data, label="Dataset")
    column_roles = infer_column_roles(dataframe)
    return {
        "dataset_name": source_name,
        "columns": column_roles,
        "summary": summarize_roles(column_roles),
    }


def health(data: DataInput) -> dict[str, Any]:
    """Calculate the FrameVitals data-health score only."""
    dataframe, source_name = _load_input(data, label="Dataset")
    dataset_profile = build_profile(dataframe)
    payload = calculate_health_score(dataframe, dataset_profile)
    return _with_dataset_name(payload, source_name)


def ml_readiness(data: DataInput) -> dict[str, Any]:
    """Calculate ML-readiness diagnostics only."""
    dataframe, source_name = _load_input(data, label="Dataset")
    dataset_profile = build_profile(dataframe)
    payload = calculate_ml_readiness(dataframe, profile=dataset_profile)
    return _with_dataset_name(payload, source_name)


def quality(
    data: DataInput,
    *,
    max_sample_rows: int = 5_000,
    max_columns: int = 100,
    max_missingness_columns: int = 25,
) -> dict[str, Any]:
    """Run practical deterministic data-quality diagnostics only."""
    dataframe, source_name = _load_input(data, label="Dataset")
    dataset_profile = build_profile(dataframe)
    column_roles = infer_column_roles(dataframe)
    payload = run_quality_diagnostics(
        dataframe,
        profile=dataset_profile,
        column_roles=column_roles,
        max_sample_rows=max_sample_rows,
        max_columns=max_columns,
        max_missingness_columns=max_missingness_columns,
    )
    return _with_dataset_name(payload, source_name)


def statistics(
    data: DataInput,
    *,
    max_pairs: int = 20,
) -> dict[str, Any]:
    """Run the deep statistical diagnostics layer only."""
    dataframe, source_name = _load_input(data, label="Dataset")
    payload = run_deep_statistics_v2(dataframe, max_pairs=max_pairs)
    return _with_dataset_name(payload, source_name)


def anomalies(
    data: DataInput,
    *,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
) -> dict[str, Any]:
    """Run the tabular anomaly ensemble only."""
    dataframe, source_name = _load_input(data, label="Dataset")
    payload = detect_anomalies_ensemble(
        dataframe,
        contamination=contamination,
        threshold=threshold,
        max_columns=max_columns,
        top_k=top_k,
    )
    return _with_dataset_name(payload, source_name)


def target_analysis(
    data: DataInput,
    *,
    target: str,
) -> dict[str, Any]:
    """Run target-quality, leakage, association, and split diagnostics only."""
    dataframe, source_name = _load_input(data, label="Dataset")
    if target not in dataframe.columns:
        raise ValueError(f"Target column not found: {target}")
    column_roles = infer_column_roles(dataframe)
    payload = run_target_intelligence(
        dataframe,
        target_column=target,
        column_roles=column_roles,
    )
    return _with_dataset_name(payload, source_name)


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
    """Analyze a tabular dataset with the complete configured FrameVitals pipeline."""
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

    pipeline_kwargs = {
        "dataset_id": dataset_id,
        "original_filename": "<dataframe>",
        "analysis_mode": resolved.mode,
        "target_column": resolved.target,
        "parallel_workers": resolved.workers,
        "skip_ai": True,
        "write_artifacts": resolved.artifacts,
        "disabled_modules": resolved.disabled_modules,
    }

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Dataset DataFrame is empty.")
        payload = run_full_analysis(
            dataframe=data,
            **pipeline_kwargs,
        )
    elif isinstance(data, (str, Path)):
        path = _validated_path(data)
        pipeline_kwargs["original_filename"] = path.name
        payload = run_full_analysis(
            file_path=path,
            **pipeline_kwargs,
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
    disabled_modules: list[str] | tuple[str, ...] | None = None,
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
        disabled_modules=disabled_modules,
    )
    dataframe, source_name = _load_input(data, label="Dataset")
    dataset_profile = build_profile(dataframe)
    column_roles = infer_column_roles(dataframe)
    dataset_signals = detect_dataset_signals(
        dataframe,
        dataset_profile,
        column_roles=column_roles,
    )
    selection = select_analyses(
        signals=dataset_signals,
        analysis_mode=resolved.mode,
        target_column=resolved.target,
    )
    disabled = set(resolved.disabled_modules)
    selection["execution_modules"] = {
        "disabled": sorted(disabled),
        "enabled": sorted(VALID_MODULES - disabled),
    }

    public_signals = {
        key: value
        for key, value in dataset_signals.items()
        if key != "column_roles"
    }
    return AnalysisPlan({
        "dataset_name": source_name,
        "analysis_mode": resolved.mode,
        "target": resolved.target,
        "shape": dict(dataset_profile.get("shape", {})),
        "config": resolved.to_dict(),
        "signals": public_signals,
        "selection": selection,
    })


def plan_cleaning(data: DataInput) -> CleaningPlan:
    """Infer a conservative cleaning plan without modifying the input data."""
    dataframe, _ = _load_input(data, label="Dataset")
    dataset_profile = build_profile(dataframe)
    return infer_cleaning_plan(dataframe, profile=dataset_profile)


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
) -> DriftResult:
    """Compare reference and current datasets for distribution and schema drift."""
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    ref_df, ref_name = _load_input(reference, label="Reference")
    cur_df, cur_name = _load_input(current, label="Current")

    payload = compare_datasets(
        ref_df,
        cur_df,
        columns=columns,
        max_columns=max_columns,
    )
    payload["reference_name"] = ref_name
    payload["current_name"] = cur_name
    return DriftResult(payload)


def infer_contract(
    data: DataInput,
    *,
    numeric_tolerance: float = 0.05,
    max_categories: int = 20,
    null_fraction_tolerance: float = 0.05,
    infer_unique: bool = True,
    min_unique_rows: int = 20,
    allow_extra_columns: bool = False,
) -> dict[str, Any]:
    """Infer a JSON-serializable data contract from a reference dataset."""
    dataframe, source_name = _load_input(data, label="Reference")
    contract = _infer_contract(
        dataframe,
        numeric_tolerance=numeric_tolerance,
        max_categories=max_categories,
        null_fraction_tolerance=null_fraction_tolerance,
        infer_unique=infer_unique,
        min_unique_rows=min_unique_rows,
        allow_extra_columns=allow_extra_columns,
    )
    contract["reference_name"] = source_name
    return contract


def validate(
    data: DataInput,
    contract: Mapping[str, Any],
) -> ValidationResult:
    """Validate a dataset against an inferred or explicit data contract."""
    dataframe, source_name = _load_input(data, label="Dataset")
    payload = validate_contract(dataframe, contract)
    payload["dataset_name"] = source_name
    return ValidationResult(payload)


def gate(
    current: DataInput,
    *,
    reference: DataInput | None = None,
    contract: Mapping[str, Any] | None = None,
    columns: list[str] | None = None,
    max_columns: int = 30,
    drift_warn_on: str = "moderate",
    drift_fail_on: str = "severe",
    fail_on_validation_warning: bool = False,
) -> GateResult:
    """Run lightweight validation/drift checks and return one quality verdict.

    At least one of ``reference`` or ``contract`` must be supplied. The current
    dataset is loaded only once, and no full EDA/model pipeline is executed.
    Validation failures always fail the gate. Warning-only validation results
    warn by default and can be promoted to failures. Drift uses independent
    warning/failure thresholds.
    """
    if reference is None and contract is None:
        raise ValueError("gate requires at least one of reference= or contract=.")
    if drift_warn_on not in _DRIFT_SEVERITY_RANK:
        raise ValueError("drift_warn_on must be one of: stable, minor, moderate, severe.")
    if drift_fail_on not in _DRIFT_SEVERITY_RANK:
        raise ValueError("drift_fail_on must be one of: stable, minor, moderate, severe.")
    if _DRIFT_SEVERITY_RANK[drift_warn_on] > _DRIFT_SEVERITY_RANK[drift_fail_on]:
        raise ValueError("drift_warn_on cannot be more severe than drift_fail_on.")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    current_df, current_name = _load_input(current, label="Current")
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    status = "pass"

    def warn() -> None:
        nonlocal status
        if status == "pass":
            status = "warn"

    def fail() -> None:
        nonlocal status
        status = "fail"

    if contract is not None:
        validation = validate_contract(current_df, contract)
        validation["dataset_name"] = current_name
        checks["validation"] = validation

        validation_status = validation.get("status")
        if validation_status == "fail":
            fail()
            error_count = validation.get("summary", {}).get("errors", 0)
            reasons.append(f"Contract validation failed with {error_count} error(s).")
        elif validation_status == "warn":
            warning_count = validation.get("summary", {}).get("warnings", 0)
            if fail_on_validation_warning:
                fail()
                reasons.append(
                    f"Contract validation produced {warning_count} warning(s), promoted to failure."
                )
            else:
                warn()
                reasons.append(f"Contract validation produced {warning_count} warning(s).")

    if reference is not None:
        reference_df, reference_name = _load_input(reference, label="Reference")
        drift_payload = compare_datasets(
            reference_df,
            current_df,
            columns=columns,
            max_columns=max_columns,
        )
        drift_payload["reference_name"] = reference_name
        drift_payload["current_name"] = current_name
        checks["drift"] = drift_payload

        if not drift_payload.get("available"):
            fail()
            reasons.append(
                "Drift comparison was requested but could not produce a comparable result: "
                f"{drift_payload.get('reason', 'unknown reason')}."
            )
        else:
            drift_severity = str(
                drift_payload.get("gate", {}).get("severity", "unknown")
            )
            if drift_severity not in _DRIFT_SEVERITY_RANK:
                fail()
                reasons.append(
                    "Drift comparison did not produce a recognized severity verdict."
                )
            elif severity_at_least(drift_severity, drift_fail_on):
                fail()
                reasons.append(
                    f"Drift severity {drift_severity} reached fail threshold {drift_fail_on}."
                )
            elif severity_at_least(drift_severity, drift_warn_on):
                warn()
                reasons.append(
                    f"Drift severity {drift_severity} reached warning threshold {drift_warn_on}."
                )

            drift_reasons = drift_payload.get("gate", {}).get("reasons", [])
            if isinstance(drift_reasons, list):
                for reason in drift_reasons[:10]:
                    text = str(reason)
                    if text and text not in reasons:
                        reasons.append(text)

    return GateResult({
        "status": status,
        "passed": status != "fail",
        "current_name": current_name,
        "checks_run": list(checks),
        "thresholds": {
            "drift_warn_on": drift_warn_on,
            "drift_fail_on": drift_fail_on,
            "fail_on_validation_warning": bool(fail_on_validation_warning),
        },
        "reasons": reasons,
        "checks": checks,
    })
