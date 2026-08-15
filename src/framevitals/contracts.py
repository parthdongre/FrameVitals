"""FrameVitals data contracts.

A contract captures conservative expectations learned from a known-good
reference dataset. Validation returns evidence-rich pass/warn/fail checks that
can be consumed in notebooks, applications, or CI pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from framevitals.column_roles import infer_column_roles


CONTRACT_VERSION = 1


def _json_value(value: Any) -> Any:
    """Convert common numpy/pandas scalar values into JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _dtype_family(series: pd.Series, roles: set[str] | None = None) -> str:
    roles = roles or set()
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if "time_like" in roles:
        return "datetime_like"
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "categorical"
    return "other"


def _matches_dtype_family(series: pd.Series, expected: str) -> tuple[bool, str]:
    if expected == "boolean":
        return bool(pd.api.types.is_bool_dtype(series)), str(series.dtype)
    if expected == "numeric":
        return bool(pd.api.types.is_numeric_dtype(series)), str(series.dtype)
    if expected == "datetime":
        return bool(pd.api.types.is_datetime64_any_dtype(series)), str(series.dtype)
    if expected == "categorical":
        matched = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        return bool(matched), str(series.dtype)
    if expected == "datetime_like":
        clean = series.dropna()
        if clean.empty:
            return True, "empty/non-null sample"
        if pd.api.types.is_datetime64_any_dtype(clean):
            return True, str(series.dtype)
        sample = clean.astype(str).head(200)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        ratio = float(parsed.notna().mean()) if len(sample) else 1.0
        return ratio >= 0.7, f"parseable_ratio={ratio:.3f}, dtype={series.dtype}"
    return True, str(series.dtype)


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    column: str | None = None,
    observed: Any = None,
    expected: Any = None,
) -> dict[str, Any]:
    return {
        "check": check_id,
        "status": status,
        "column": column,
        "message": message,
        "observed": _json_value(observed),
        "expected": _json_value(expected),
    }


def infer_contract(
    df: pd.DataFrame,
    *,
    source_name: str = "<dataframe>",
    missing_tolerance: float = 0.05,
    duplicate_tolerance: float = 0.02,
    max_allowed_values: int = 20,
) -> dict[str, Any]:
    """Infer a conservative data-health contract from a reference DataFrame.

    The inferred rules intentionally avoid hard numeric min/max ranges. They
    focus on schema shape, dtype families, missingness budgets, duplicate
    budgets, ID-like uniqueness, and low-cardinality category drift.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("Cannot infer a contract from an empty DataFrame.")
    if not 0 <= missing_tolerance <= 1:
        raise ValueError("missing_tolerance must be between 0 and 1.")
    if not 0 <= duplicate_tolerance <= 1:
        raise ValueError("duplicate_tolerance must be between 0 and 1.")
    if max_allowed_values < 0:
        raise ValueError("max_allowed_values must be non-negative.")

    roles = infer_column_roles(df)
    columns: dict[str, dict[str, Any]] = {}

    for name in df.columns:
        series = df[name]
        info = roles[name]
        role_set = set(info.get("roles", []))
        missing_fraction = float(series.isna().mean())
        non_missing = series.dropna()

        allowed_values = None
        if (
            max_allowed_values > 0
            and "low_cardinality" in role_set
            and "id_like" not in role_set
            and "time_like" not in role_set
            and int(info.get("unique_count", 0)) <= max_allowed_values
        ):
            allowed_values = sorted(
                {str(_json_value(value)) for value in non_missing.unique()}
            )

        unique_required = bool(
            "id_like" in role_set
            and float(info.get("unique_ratio", 0.0)) >= 0.95
            and int(info.get("non_missing_count", 0)) >= 10
        )

        columns[name] = {
            "dtype_family": _dtype_family(series, role_set),
            "baseline_dtype": str(series.dtype),
            "required": True,
            "max_missing_fraction": round(
                min(1.0, missing_fraction + missing_tolerance),
                4,
            ),
            "baseline_missing_fraction": round(missing_fraction, 4),
            "unique": unique_required,
            "allowed_values": allowed_values,
            "allowed_values_policy": "warn" if allowed_values is not None else "ignore",
            "roles": sorted(role_set),
        }

    duplicate_fraction = float(df.duplicated().mean())

    return {
        "contract_version": CONTRACT_VERSION,
        "generated_by": "framevitals",
        "source_name": source_name,
        "baseline": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "duplicate_fraction": round(duplicate_fraction, 4),
        },
        "policies": {
            "extra_columns": "warn",
            "max_duplicate_fraction": round(
                min(1.0, duplicate_fraction + duplicate_tolerance),
                4,
            ),
        },
        "columns": columns,
    }


def validate_contract(
    df: pd.DataFrame,
    contract: dict[str, Any],
    *,
    source_name: str = "<dataframe>",
) -> dict[str, Any]:
    """Validate a DataFrame against a FrameVitals contract."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(contract, dict):
        raise TypeError("contract must be a dictionary")
    if df.empty:
        raise ValueError("Cannot validate an empty DataFrame.")

    version = contract.get("contract_version")
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported contract_version {version!r}; expected {CONTRACT_VERSION}."
        )

    column_rules = contract.get("columns")
    if not isinstance(column_rules, dict) or not column_rules:
        raise ValueError("Contract does not contain any column rules.")

    checks: list[dict[str, Any]] = []
    expected_columns = set(column_rules)
    actual_columns = set(df.columns)

    missing_columns = sorted(expected_columns - actual_columns)
    extra_columns = sorted(actual_columns - expected_columns)

    for column in missing_columns:
        rule = column_rules[column]
        if rule.get("required", True):
            checks.append(
                _check(
                    "required_column",
                    "fail",
                    f"Required column '{column}' is missing.",
                    column=column,
                    observed=False,
                    expected=True,
                )
            )

    extra_policy = contract.get("policies", {}).get("extra_columns", "warn")
    if extra_columns and extra_policy != "ignore":
        status = "fail" if extra_policy == "fail" else "warn"
        checks.append(
            _check(
                "extra_columns",
                status,
                f"Found {len(extra_columns)} unexpected column(s): {', '.join(extra_columns[:10])}.",
                observed=extra_columns,
                expected="no unexpected columns" if extra_policy == "fail" else "review",
            )
        )

    for column, rule in column_rules.items():
        if column not in df.columns:
            continue

        series = df[column]
        expected_family = str(rule.get("dtype_family", "other"))
        matched, dtype_evidence = _matches_dtype_family(series, expected_family)
        checks.append(
            _check(
                "dtype_family",
                "pass" if matched else "fail",
                (
                    f"Column '{column}' matches expected dtype family '{expected_family}'."
                    if matched
                    else f"Column '{column}' does not match expected dtype family '{expected_family}'."
                ),
                column=column,
                observed=dtype_evidence,
                expected=expected_family,
            )
        )

        missing_fraction = float(series.isna().mean())
        max_missing = float(rule.get("max_missing_fraction", 1.0))
        missing_ok = missing_fraction <= max_missing + 1e-12
        checks.append(
            _check(
                "missing_fraction",
                "pass" if missing_ok else "fail",
                (
                    f"Column '{column}' missingness is within its budget."
                    if missing_ok
                    else f"Column '{column}' exceeds its missingness budget."
                ),
                column=column,
                observed=round(missing_fraction, 4),
                expected=f"<= {max_missing:.4f}",
            )
        )

        if rule.get("unique"):
            non_missing = series.dropna()
            duplicate_values = int(non_missing.duplicated().sum())
            unique_ok = duplicate_values == 0
            checks.append(
                _check(
                    "unique",
                    "pass" if unique_ok else "fail",
                    (
                        f"Column '{column}' remains unique."
                        if unique_ok
                        else f"Column '{column}' contains {duplicate_values} duplicate non-null value(s)."
                    ),
                    column=column,
                    observed=duplicate_values,
                    expected=0,
                )
            )

        allowed_values = rule.get("allowed_values")
        allowed_policy = rule.get("allowed_values_policy", "ignore")
        if allowed_values is not None and allowed_policy != "ignore":
            allowed = {str(value) for value in allowed_values}
            observed_values = {str(_json_value(value)) for value in series.dropna().unique()}
            unseen = sorted(observed_values - allowed)
            if unseen:
                status = "fail" if allowed_policy == "fail" else "warn"
                checks.append(
                    _check(
                        "allowed_values",
                        status,
                        f"Column '{column}' contains {len(unseen)} unseen value(s).",
                        column=column,
                        observed=unseen[:20],
                        expected=sorted(allowed)[:20],
                    )
                )
            else:
                checks.append(
                    _check(
                        "allowed_values",
                        "pass",
                        f"Column '{column}' contains no unseen categorical values.",
                        column=column,
                        observed="no unseen values",
                        expected="baseline categories",
                    )
                )

    duplicate_fraction = float(df.duplicated().mean())
    max_duplicate_fraction = float(
        contract.get("policies", {}).get("max_duplicate_fraction", 1.0)
    )
    duplicate_ok = duplicate_fraction <= max_duplicate_fraction + 1e-12
    checks.append(
        _check(
            "duplicate_fraction",
            "pass" if duplicate_ok else "fail",
            (
                "Dataset duplicate rate is within its budget."
                if duplicate_ok
                else "Dataset duplicate rate exceeds its budget."
            ),
            observed=round(duplicate_fraction, 4),
            expected=f"<= {max_duplicate_fraction:.4f}",
        )
    )

    counts = {
        status: sum(1 for item in checks if item["status"] == status)
        for status in ("pass", "warn", "fail")
    }
    if counts["fail"]:
        overall_status = "fail"
    elif counts["warn"]:
        overall_status = "warn"
    else:
        overall_status = "pass"

    return {
        "available": True,
        "source_name": source_name,
        "contract_source": contract.get("source_name"),
        "contract_version": version,
        "status": overall_status,
        "valid": counts["fail"] == 0,
        "summary": counts,
        "schema": {
            "expected_columns": len(expected_columns),
            "actual_columns": int(df.shape[1]),
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
        },
        "checks": checks,
    }


def load_contract(path: str | Path) -> dict[str, Any]:
    """Load a JSON FrameVitals contract from disk."""
    contract_path = Path(path)
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract not found: {contract_path}")
    if not contract_path.is_file():
        raise ValueError(f"Expected a contract file, got: {contract_path}")
    try:
        value = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON contract: {contract_path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Contract JSON must contain an object at the top level.")
    return value


def save_contract(contract: dict[str, Any], path: str | Path) -> Path:
    """Persist a contract as deterministic, human-readable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(contract, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return target
