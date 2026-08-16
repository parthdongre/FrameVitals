"""Lightweight public data operations that do not require the full pipeline.

Cleaning, contracts, drift, and quality gates are useful independently of EDA,
modeling, explainability, and report generation. Keeping them here prevents a
simple validation or comparison call from importing the heavy analysis stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from framevitals.profiler import build_profile
from framevitals.quality_results import DriftResult, GateResult, ValidationResult
from framevitals.sources import DatasetMetadata, StreamingDatasetSource, resolve_source


DataInput = str | Path | pd.DataFrame
_DRIFT_SEVERITY_RANK = {"stable": 0, "minor": 1, "moderate": 2, "severe": 3}
_DRIFT_SAMPLE_ROWS = 50_000


def _resolve_input(value: DataInput, *, label: str):
    try:
        source = resolve_source(value)
        metadata = source.inspect()
    except (TypeError, ValueError, FileNotFoundError) as exc:
        if label == "Dataset":
            raise
        message = str(exc).replace("Dataset", label, 1)
        raise type(exc)(message) from exc

    if metadata.rows == 0:
        if metadata.kind == "memory":
            raise ValueError(f"{label} DataFrame is empty.")
        raise ValueError(f"{label} dataset is empty: {metadata.name}")
    return source, metadata


def _load_input(value: DataInput, *, label: str) -> tuple[pd.DataFrame, str]:
    source, metadata = _resolve_input(value, label=label)
    dataframe = source.load()
    if dataframe.empty:
        if metadata.kind == "memory":
            raise ValueError(f"{label} DataFrame is empty.")
        raise ValueError(f"{label} dataset is empty: {metadata.name}")
    return dataframe, metadata.name


def _comparison_frame(
    source,
    metadata: DatasetMetadata,
    *,
    sample_rows: int = _DRIFT_SAMPLE_ROWS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return drift input plus transparent source/materialization metadata."""
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import sample_streaming_source

        if metadata.rows is None:
            raise ValueError("Streaming drift comparison requires a source row count.")
        sample = sample_streaming_source(source, sample_rows=sample_rows)
        source_rows = int(metadata.rows)
        sampled = len(sample) < source_rows
        return sample, {
            "source_rows": source_rows,
            "source_columns": int(metadata.columns or len(sample.columns)),
            "sample_rows": int(len(sample)),
            "sampled": sampled,
            "strategy": (
                "streaming_evenly_spaced_global_rows"
                if sampled
                else "full_stream_via_batches"
            ),
            "full_materialization": False,
            "source": metadata.to_dict(),
        }

    dataframe = source.load()
    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {metadata.name}")
    return dataframe, {
        "source_rows": int(len(dataframe)),
        "source_columns": int(len(dataframe.columns)),
        "sample_rows": int(len(dataframe)),
        "sampled": False,
        "strategy": "full_input",
        "full_materialization": bool(metadata.kind == "file"),
        "source": metadata.to_dict(),
    }


def _true_shape(execution: Mapping[str, Any]) -> list[int]:
    return [
        int(execution.get("source_rows", 0)),
        int(execution.get("source_columns", 0)),
    ]


def _row_count_change_percent(reference_rows: int, current_rows: int) -> float | None:
    if reference_rows <= 0:
        return None
    return round((current_rows - reference_rows) / reference_rows * 100, 4)


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
    """Compare datasets, bounding value-distribution work for streaming sources."""
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    reference_source, reference_metadata = _resolve_input(reference, label="Reference")
    current_source, current_metadata = _resolve_input(current, label="Current")
    reference_df, reference_execution = _comparison_frame(
        reference_source,
        reference_metadata,
    )
    current_df, current_execution = _comparison_frame(
        current_source,
        current_metadata,
    )

    payload = compare_datasets(
        reference_df,
        current_df,
        columns=columns,
        max_columns=max_columns,
    )
    payload["reference_name"] = reference_metadata.name
    payload["current_name"] = current_metadata.name
    payload["ref_shape"] = _true_shape(reference_execution)
    payload["cur_shape"] = _true_shape(current_execution)
    payload["row_count_change_percent"] = _row_count_change_percent(
        reference_execution["source_rows"],
        current_execution["source_rows"],
    )

    any_sampled = bool(
        reference_execution["sampled"] or current_execution["sampled"]
    )
    payload["execution"] = {
        "method": "bounded_source_compare" if any_sampled else "full_compare",
        "sample_limit_rows_per_source": _DRIFT_SAMPLE_ROWS,
        "full_materialization": bool(
            reference_execution["full_materialization"]
            or current_execution["full_materialization"]
        ),
        "reference": reference_execution,
        "current": current_execution,
        "components": {
            "source_shape": "exact",
            "schema_columns": "exact",
            "value_distributions": (
                "bounded_row_sample" if any_sampled else "full_input"
            ),
            "missingness": "bounded_row_sample" if any_sampled else "full_input",
        },
    }
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
    source, metadata = _resolve_input(data, label="Dataset")
    dataframe = source.load()
    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {metadata.name}")
    payload = validate_contract(dataframe, contract)
    payload["dataset_name"] = metadata.name
    payload["execution"] = {
        "method": "exact_contract_validation",
        "full_materialization": bool(metadata.kind == "file"),
        "source": metadata.to_dict(),
        "reason": (
            "Contract validation remains exact; uniqueness, allowed-value, and bound "
            "constraints are not silently downgraded to sampled checks."
        ),
    }
    return ValidationResult(payload)


def gate(
    current: DataInput,
    *,
    reference: DataInput | None = None,
    contract: Mapping[str, Any] | None = None,
    custom_checks: Sequence[Any] | None = None,
    columns: list[str] | None = None,
    max_columns: int = 30,
    drift_warn_on: str = "moderate",
    drift_fail_on: str = "severe",
    fail_on_validation_warning: bool = False,
) -> GateResult:
    """Combine contracts, custom invariants, and bounded drift into one verdict."""
    if reference is None and contract is None and not custom_checks:
        raise ValueError(
            "gate requires at least one of reference=, contract=, or custom_checks=."
        )
    if drift_warn_on not in _DRIFT_SEVERITY_RANK:
        raise ValueError("drift_warn_on must be one of: stable, minor, moderate, severe.")
    if drift_fail_on not in _DRIFT_SEVERITY_RANK:
        raise ValueError("drift_fail_on must be one of: stable, minor, moderate, severe.")
    if _DRIFT_SEVERITY_RANK[drift_warn_on] > _DRIFT_SEVERITY_RANK[drift_fail_on]:
        raise ValueError("drift_warn_on cannot be more severe than drift_fail_on.")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")

    _, current_metadata = _resolve_input(current, label="Current")
    current_name = current_metadata.name
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
        validation_payload = validate(current, contract)
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
        drift_payload = compare(
            reference,
            current,
            columns=columns,
            max_columns=max_columns,
        )
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

    if custom_checks:
        from framevitals.checks import run_checks

        custom_payload = run_checks(current, custom_checks)
        checks["custom"] = custom_payload
        custom_status = custom_payload.get("status")
        if custom_status == "fail":
            fail()
        elif custom_status == "warn":
            warn()

        for result in custom_payload.get("results", [])[:10]:
            if not isinstance(result, Mapping) or result.get("passed"):
                continue
            message = str(result.get("message") or result.get("name") or "Custom check failed.")
            if message and message not in reasons:
                reasons.append(message)

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
        "execution": {
            "validation": (
                checks.get("validation", {}).get("execution")
                if isinstance(checks.get("validation"), Mapping)
                else None
            ),
            "drift": (
                checks.get("drift", {}).get("execution")
                if isinstance(checks.get("drift"), Mapping)
                else None
            ),
            "custom": (
                checks.get("custom", {}).get("execution")
                if isinstance(checks.get("custom"), Mapping)
                else None
            ),
        },
    })
