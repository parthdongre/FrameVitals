"""Inference and validation helpers for lightweight data contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

CONTRACT_VERSION = 1
_TYPE_FAMILIES = {
    "boolean",
    "datetime",
    "integer",
    "number",
    "string",
}


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


def _numeric_bounds(series: pd.Series) -> dict[str, int | float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return {}

    minimum = values.min()
    maximum = values.max()
    return {
        "minimum": minimum.item() if hasattr(minimum, "item") else minimum,
        "maximum": maximum.item() if hasattr(maximum, "item") else maximum,
    }


def infer_contract(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Create a JSON-serializable contract from a reference dataset.

    The inferred contract captures schema and basic quality constraints that
    are stable enough to use as an automated data gate: required columns,
    broad type families, nullability, and finite numeric bounds.
    """
    _require_contractible_columns(dataframe)

    columns: dict[str, dict[str, Any]] = {}
    for name in dataframe.columns:
        series = dataframe[name]
        family = _type_family(series)
        specification: dict[str, Any] = {
            "type": family,
            "nullable": bool(series.isna().any()),
        }
        if family in {"integer", "number"}:
            specification.update(_numeric_bounds(series))
        columns[name] = specification

    return {
        "version": CONTRACT_VERSION,
        "allow_extra_columns": False,
        "columns": columns,
    }


def _normalise_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TypeError("contract must be a mapping produced by infer_contract().")

    columns = contract.get("columns")
    if not isinstance(columns, Mapping) or not columns:
        raise ValueError("contract must contain a non-empty 'columns' mapping.")

    version = contract.get("version", CONTRACT_VERSION)
    if version != CONTRACT_VERSION:
        raise ValueError(f"Unsupported contract version: {version!r}.")

    allow_extra_columns = contract.get("allow_extra_columns", False)
    if not isinstance(allow_extra_columns, bool):
        raise ValueError("contract field 'allow_extra_columns' must be a boolean.")

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

        nullable = specification.get("nullable", True)
        if not isinstance(nullable, bool):
            raise ValueError(f"contract field 'nullable' for '{name}' must be a boolean.")

        normalised = {
            "type": family,
            "nullable": nullable,
        }
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
        normalised_columns[name] = normalised

    return {
        "version": version,
        "allow_extra_columns": allow_extra_columns,
        "columns": normalised_columns,
    }


def _compatible_type(expected: str, actual: str) -> bool:
    if expected == "number":
        return actual in {"integer", "number"}
    return expected == actual


def _finding(code: str, column: str, message: str, **details: Any) -> dict[str, Any]:
    finding: dict[str, Any] = {
        "code": code,
        "column": column,
        "message": message,
    }
    finding.update(details)
    return finding


def validate_contract(dataframe: pd.DataFrame, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a dataset against an inferred or explicit data contract."""
    _require_contractible_columns(dataframe)
    normalised = _normalise_contract(contract)
    specifications = normalised["columns"]
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    actual_columns = set(dataframe.columns)
    expected_columns = set(specifications)
    for name in sorted(expected_columns - actual_columns):
        errors.append(
            _finding(
                "missing_column",
                name,
                f"Required column '{name}' is missing.",
            )
        )

    if not normalised["allow_extra_columns"]:
        for name in sorted(actual_columns - expected_columns):
            errors.append(
                _finding(
                    "unexpected_column",
                    name,
                    f"Column '{name}' is not defined by the contract.",
                )
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
        if not specification["nullable"] and null_count:
            errors.append(
                _finding(
                    "nullability_violation",
                    name,
                    f"Column '{name}' does not allow null values ({null_count} found).",
                    null_count=null_count,
                )
            )

        if expected_type not in {"integer", "number"}:
            continue

        values = pd.to_numeric(series, errors="coerce").dropna()
        values = values[np.isfinite(values)]
        if values.empty:
            continue

        observed_minimum = values.min()
        observed_maximum = values.max()
        observed_minimum = (
            observed_minimum.item()
            if hasattr(observed_minimum, "item")
            else observed_minimum
        )
        observed_maximum = (
            observed_maximum.item()
            if hasattr(observed_maximum, "item")
            else observed_maximum
        )

        minimum = specification.get("minimum")
        if minimum is not None and observed_minimum < minimum:
            errors.append(
                _finding(
                    "minimum_violation",
                    name,
                    f"Column '{name}' contains values below its minimum of {minimum}.",
                    minimum=minimum,
                    observed_minimum=observed_minimum,
                )
            )

        maximum = specification.get("maximum")
        if maximum is not None and observed_maximum > maximum:
            errors.append(
                _finding(
                    "maximum_violation",
                    name,
                    f"Column '{name}' contains values above its maximum of {maximum}.",
                    maximum=maximum,
                    observed_maximum=observed_maximum,
                )
            )

    return {
        "valid": not errors,
        "contract_version": normalised["version"],
        "data": {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
        },
        "summary": {
            "columns_checked": len(specifications),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
    }
