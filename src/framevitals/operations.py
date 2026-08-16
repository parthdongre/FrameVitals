"""Lightweight public data operations that do not require the full pipeline.

Cleaning, contracts, drift, and quality gates are useful independently of EDA,
modeling, explainability, and report generation. Keeping them here prevents a
simple validation or comparison call from importing the heavy analysis stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from framevitals.cleaning_plan import (
    CleaningPlan,
    apply_cleaning_plan,
    infer_cleaning_plan,
)
from framevitals.contracts import infer_contract as _infer_contract
from framevitals.contracts import validate_contract
from framevitals.drift_analysis import compare_datasets, severity_at_least
from framevitals.loader import load_dataset
from framevitals.profiler import build_profile
from framevitals.quality_results import DriftResult, GateResult, ValidationResult


DataInput = str | Path | pd.DataFrame
_DRIFT_SEVERITY_RANK = {"stable": 0, "minor": 1, "moderate": 2, "severe": 3}


def _validated_path(value: str | Path, *, label: str) -> Path:
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
        dataframe = load_dataset(path)
        if dataframe.empty:
            raise ValueError(f"{label} dataset is empty: {path}")
        return dataframe, path.name

    raise TypeError(
        f"{label} must be a pandas DataFrame or a path to a supported dataset."
    )


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
    """Return an explicitly cleaned copy; never mutate the caller's input."""
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
    """Compare reference/current datasets without executing the EDA pipeline."""
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    reference_df, reference_name = _load_input(reference, label="Reference")
    current_df, current_name = _load_input(current, label="Current")
    payload = compare_datasets(
        reference_df,
        current_df,
        columns=columns,
        max_columns=max_columns,
    )
    payload["reference_name"] = reference_name
    payload["current_name"] = current_name
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
    """Infer a JSON-serializable contract from a reference dataset."""
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
    """Validate a dataset against an inferred or explicit contract."""
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
    """Combine contract/drift checks into one CI-friendly quality verdict."""
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
        validation_payload = validate_contract(current_df, contract)
        validation_payload["dataset_name"] = current_name
        checks["validation"] = validation_payload

        validation_status = validation_payload.get("status")
        if validation_status == "fail":
            fail()
            error_count = validation_payload.get("summary", {}).get("errors", 0)
            reasons.append(f"Contract validation failed with {error_count} error(s).")
        elif validation_status == "warn":
            warning_count = validation_payload.get("summary", {}).get("warnings", 0)
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
