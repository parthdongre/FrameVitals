"""Inference and validation helpers for reusable data contracts.

Contracts are intentionally lightweight JSON-compatible dictionaries. Version 2
adds tolerant numeric bounds, optional columns, low-cardinality value sets,
null-rate ceilings, uniqueness hints, and normalized validation severities while
remaining able to validate version-1 contracts created by earlier FrameVitals
builds.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

CONTRACT_VERSION = 2
_SUPPORTED_CONTRACT_VERSIONS = {1, 2}
_TYPE_FAMILIES = {
    "boolean",
    "datetime",
    "integer",
    "number",
    "string",
}
_SEVERITIES = {"ignore", "warning", "error"}


def _require_contractible_columns(dataframe: pd.DataFrame) -> None:
    if not dataframe.columns.is_unique:
        raise ValueError("Data contracts require unique column names.")

    non_string_columns = [
        column
        for column in dataframe.columns
        if not isinstance(column, str)
    ]
    if non_string_columns:
        raise ValueError("Data contracts require string column names.")


def _type_family(series: pd.Series) -> str:
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    if pdt.is_integer_dtype(series):
        return "integer"
    if pdt.is_numeric_dtype(series):
        return "number"
    return "string"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _numeric_bounds(
    series: pd.Series,
    *,
    tolerance: float,
    family: str,
) -> dict[str, int | float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return {}

    observed_min = float(values.min())
    observed_max = float(values.max())
    span = observed_max - observed_min
    if span <= 0:
        span = max(abs(observed_min), 1.0)
    margin = span * tolerance

    minimum = observed_min - margin
    maximum = observed_max + margin
    if family == "integer":
        minimum = int(np.floor(minimum))
        maximum = int(np.ceil(maximum))

    return {
        "minimum": _json_scalar(minimum),
        "maximum": _json_scalar(maximum),
        "observed_minimum": _json_scalar(observed_min),
        "observed_maximum": _json_scalar(observed_max),
    }


def _allowed_values(series: pd.Series, *, max_categories: int) -> list[Any] | None:
    clean = series.dropna()
    if clean.empty:
        return None
    unique = clean.unique()
    if len(unique) > max_categories:
        return None
    values = [_json_scalar(value) for value in unique.tolist()]
    return sorted(values, key=lambda value: str(value))


def infer_contract(
    dataframe: pd.DataFrame,
    *,
    numeric_tolerance: float = 0.05,
    max_categories: int = 20,
    null_fraction_tolerance: float = 0.05,
    infer_unique: bool = True,
    min_unique_rows: int = 20,
    allow_extra_columns: bool = False,
) -> dict[str, Any]:
    """Create a JSON-serializable contract from a reference dataset.

    Inference is deliberately conservative. Numeric extrema are expanded by a
    configurable tolerance so ordinary future observations do not fail merely
    because they exceed a reference sample's exact minimum or maximum. Small
    categorical domains are recorded as warning-level expectations. Fully
    unique columns are marked unique only when enough rows exist to make that
    inference meaningful.
    """
    _require_contractible_columns(dataframe)

    if numeric_tolerance < 0:
        raise ValueError("numeric_tolerance must be non-negative.")
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1.")
    if not 0 <= null_fraction_tolerance <= 1:
        raise ValueError("null_fraction_tolerance must be between 0 and 1.")
    if min_unique_rows < 2:
        raise ValueError("min_unique_rows must be at least 2.")

    columns: dict[str, dict[str, Any]] = {}
    for name in dataframe.columns:
        series = dataframe[name]
        family = _type_family(series)
        null_fraction = float(series.isna().mean()) if len(series) else 0.0
        clean = series.dropna()

        specification: dict[str, Any] = {
            "type": family,
            "required": True,
            "nullable": bool(series.isna().any()),
            "max_null_fraction": round(
                min(1.0, null_fraction + null_fraction_tolerance),
                6,
            ),
        }

        if family in {"integer", "number"}:
            specification.update(
                _numeric_bounds(
                    series,
                    tolerance=numeric_tolerance,
                    family=family,
                )
            )
        elif family in {"string", "boolean"}:
            allowed = _allowed_values(series, max_categories=max_categories)
            if allowed is not None:
                specification["allowed_values"] = allowed
                specification["allowed_values_severity"] = "warning"

        if (
            infer_unique
            and len(clean) >= min_unique_rows
            and int(clean.nunique(dropna=True)) == len(clean)
        ):
            specification["unique"] = True

        columns[name] = specification

    return {
        "version": CONTRACT_VERSION,
        "allow_extra_columns": bool(allow_extra_columns),
        "extra_columns_severity": "ignore" if allow_extra_columns else "error",
        "inference": {
            "numeric_tolerance": float(numeric_tolerance),
            "max_categories": int(max_categories),
            "null_fraction_tolerance": float(null_fraction_tolerance),
            "infer_unique": bool(infer_unique),
            "min_unique_rows": int(min_unique_rows),
        },
        "columns": columns,
    }


def _normalise_severity(value: Any, *, field: str) -> str:
    severity = str(value).strip().lower()
    if severity not in _SEVERITIES:
        choices = ", ".join(sorted(_SEVERITIES))
        raise ValueError(f"contract field '{field}' must be one of: {choices}.")
    return severity


def _normalise_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TypeError("contract must be a mapping produced by infer_contract().")

    columns = contract.get("columns")
    if not isinstance(columns, Mapping) or not columns:
        raise ValueError("contract must contain a non-empty 'columns' mapping.")

    version = contract.get("version", 1)
    if version not in _SUPPORTED_CONTRACT_VERSIONS:
        raise ValueError(f"Unsupported contract version: {version!r}.")

    allow_extra_columns = contract.get("allow_extra_columns", False)
    if not isinstance(allow_extra_columns, bool):
        raise ValueError("contract field 'allow_extra_columns' must be a boolean.")

    default_extra_severity = "ignore" if allow_extra_columns else "error"
    extra_columns_severity = _normalise_severity(
        contract.get("extra_columns_severity", default_extra_severity),
        field="extra_columns_severity",
    )

    normalised_columns: dict[str, dict[str, Any]] = {}
    for name, specification in columns.items():
        if not isinstance(name, str):
            raise ValueError("contract column names must be strings.")
        if not isinstance(specification, Mapping):
            raise ValueError(f"contract specification for '{name}' must be a mapping.")

        family = specification.get("type")
        if family not in _TYPE_FAMILIES:
            choices = ", ".join(sorted(_TYPE_FAMILIES))
            raise ValueError(f"contract type for '{name}' must be one of: {choices}.")

        required = specification.get("required", True)
        nullable = specification.get("nullable", True)
        unique = specification.get("unique", False)
        for field_name, field_value in {
            "required": required,
            "nullable": nullable,
            "unique": unique,
        }.items():
            if not isinstance(field_value, bool):
                raise ValueError(
                    f"contract field '{field_name}' for '{name}' must be a boolean."
                )

        normalised: dict[str, Any] = {
            "type": family,
            "required": required,
            "nullable": nullable,
            "unique": unique,
        }

        max_null_fraction = specification.get("max_null_fraction")
        if max_null_fraction is not None:
            if not isinstance(max_null_fraction, int | float) or not 0 <= max_null_fraction <= 1:
                raise ValueError(
                    f"contract field 'max_null_fraction' for '{name}' must be between 0 and 1."
                )
            normalised["max_null_fraction"] = float(max_null_fraction)

        for bound in ("minimum", "maximum"):
            value = specification.get(bound)
            if value is not None:
                if family not in {"integer", "number"}:
                    raise ValueError(f"'{bound}' is only valid for numeric column '{name}'.")
                if not isinstance(value, int | float) or not np.isfinite(value):
                    raise ValueError(f"contract field '{bound}' for '{name}' must be finite.")
                normalised[bound] = value

        lower = normalised.get("minimum")
        upper = normalised.get("maximum")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"contract minimum exceeds maximum for '{name}'.")

        allowed_values = specification.get("allowed_values")
        if allowed_values is not None:
            if family not in {"string", "boolean", "integer", "number"}:
                raise ValueError(
                    f"'allowed_values' is not supported for column '{name}' of type '{family}'."
                )
            if not isinstance(allowed_values, list) or not allowed_values:
                raise ValueError(
                    f"contract field 'allowed_values' for '{name}' must be a non-empty list."
                )
            normalised["allowed_values"] = allowed_values
            normalised["allowed_values_severity"] = _normalise_severity(
                specification.get("allowed_values_severity", "error"),
                field=f"allowed_values_severity.{name}",
            )

        normalised_columns[name] = normalised

    return {
        "version": version,
        "allow_extra_columns": allow_extra_columns,
        "extra_columns_severity": extra_columns_severity,
        "columns": normalised_columns,
    }


def _compatible_type(expected: str, actual: str) -> bool:
    if expected == "number":
        return actual in {"integer", "number"}
    return expected == actual


def _finding(
    code: str,
    column: str,
    message: str,
    *,
    severity: str = "error",
    **details: Any,
) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "code": code,
        "column": column,
        "severity": severity,
        "message": message,
    }
    finding.update(details)
    return finding


def _append_finding(
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    finding: dict[str, Any],
) -> None:
    severity = finding.get("severity", "error")
    if severity == "ignore":
        return
    if severity == "warning":
        warnings.append(finding)
    else:
        errors.append(finding)


def validate_contract(dataframe: pd.DataFrame, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a dataset against an inferred or explicit data contract.

    Validation is aggregated rather than fail-fast: every applicable violation
    is collected so CI, notebooks, and users can fix a dataset in one pass.
    ``valid`` remains ``True`` when only warnings are present; ``status``
    distinguishes ``pass``, ``warn``, and ``fail``.
    """
    _require_contractible_columns(dataframe)
    normalised = _normalise_contract(contract)
    specifications = normalised["columns"]
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    actual_columns = set(dataframe.columns)
    expected_columns = set(specifications)
    for name in sorted(expected_columns - actual_columns):
        specification = specifications[name]
        if specification.get("required", True):
            errors.append(
                _finding(
                    "missing_column",
                    name,
                    f"Required column '{name}' is missing.",
                )
            )

    extra_severity = normalised["extra_columns_severity"]
    for name in sorted(actual_columns - expected_columns):
        _append_finding(
            errors,
            warnings,
            _finding(
                "unexpected_column",
                name,
                f"Column '{name}' is not defined by the contract.",
                severity=extra_severity,
            ),
        )

    for name in sorted(expected_columns & actual_columns):
        specification = specifications[name]
        series = dataframe[name]
        actual_type = _type_family(series)
        expected_type = specification["type"]

        if not _compatible_type(expected_type, actual_type):
            errors.append(
                _finding(
                    "incompatible_type",
                    name,
                    f"Column '{name}' has type '{actual_type}', expected '{expected_type}'.",
                    expected_type=expected_type,
                    actual_type=actual_type,
                )
            )
            continue

        null_count = int(series.isna().sum())
        null_fraction = float(series.isna().mean()) if len(series) else 0.0
        if not specification["nullable"] and null_count:
            errors.append(
                _finding(
                    "nullability_violation",
                    name,
                    f"Column '{name}' does not allow null values ({null_count} found).",
                    null_count=null_count,
                    null_fraction=round(null_fraction, 6),
                )
            )
        else:
            max_null_fraction = specification.get("max_null_fraction")
            if max_null_fraction is not None and null_fraction > max_null_fraction:
                warnings.append(
                    _finding(
                        "null_fraction_violation",
                        name,
                        (
                            f"Column '{name}' has {null_fraction:.1%} null values, above "
                            f"its expected ceiling of {max_null_fraction:.1%}."
                        ),
                        severity="warning",
                        null_fraction=round(null_fraction, 6),
                        max_null_fraction=max_null_fraction,
                    )
                )

        if specification.get("unique"):
            clean = series.dropna()
            duplicate_count = int(clean.duplicated().sum())
            if duplicate_count:
                errors.append(
                    _finding(
                        "uniqueness_violation",
                        name,
                        f"Column '{name}' must be unique ({duplicate_count} duplicate values found).",
                        duplicate_count=duplicate_count,
                    )
                )

        allowed_values = specification.get("allowed_values")
        if allowed_values is not None:
            allowed = set(allowed_values)
            observed = series.dropna().unique().tolist()
            unexpected_values = [
                _json_scalar(value)
                for value in observed
                if _json_scalar(value) not in allowed
            ]
            if unexpected_values:
                severity = specification.get("allowed_values_severity", "error")
                _append_finding(
                    errors,
                    warnings,
                    _finding(
                        "allowed_values_violation",
                        name,
                        f"Column '{name}' contains values outside its expected domain.",
                        severity=severity,
                        unexpected_values=unexpected_values[:20],
                        unexpected_count=len(unexpected_values),
                    ),
                )

        if expected_type not in {"integer", "number"}:
            continue

        values = pd.to_numeric(series, errors="coerce").dropna()
        values = values[np.isfinite(values)]
        if values.empty:
            continue

        observed_minimum = _json_scalar(values.min())
        observed_maximum = _json_scalar(values.max())

        minimum = specification.get("minimum")
        if minimum is not None:
            below_count = int((values < minimum).sum())
            if below_count:
                errors.append(
                    _finding(
                        "minimum_violation",
                        name,
                        f"Column '{name}' contains {below_count} values below its minimum of {minimum}.",
                        minimum=minimum,
                        observed_minimum=observed_minimum,
                        violation_count=below_count,
                    )
                )

        maximum = specification.get("maximum")
        if maximum is not None:
            above_count = int((values > maximum).sum())
            if above_count:
                errors.append(
                    _finding(
                        "maximum_violation",
                        name,
                        f"Column '{name}' contains {above_count} values above its maximum of {maximum}.",
                        maximum=maximum,
                        observed_maximum=observed_maximum,
                        violation_count=above_count,
                    )
                )

    all_findings = errors + warnings
    code_counts = dict(Counter(item["code"] for item in all_findings))
    failed_columns = sorted({item["column"] for item in errors})
    warning_columns = sorted({item["column"] for item in warnings})

    if errors:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "pass"

    return {
        "valid": not errors,
        "status": status,
        "contract_version": normalised["version"],
        "data": {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
        },
        "summary": {
            "columns_checked": len(specifications),
            "errors": len(errors),
            "warnings": len(warnings),
            "findings": len(all_findings),
            "failed_columns": failed_columns,
            "warning_columns": warning_columns,
            "code_counts": code_counts,
        },
        "errors": errors,
        "warnings": warnings,
        "findings": all_findings,
    }
