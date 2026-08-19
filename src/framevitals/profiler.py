from __future__ import annotations

import os

import numpy as np
import pandas as pd

from framevitals.backends import numeric_profile, resolve_numeric_backend
from framevitals.execution import _deterministic_stratified_positions

MAX_CORRELATION_COLUMNS = 100
MAX_EXACT_DUPLICATE_CELLS = 50_000_000
DUPLICATE_SAMPLE_ROWS = 50_000
NATIVE_NUMERIC_PROFILE_MIN_ROWS = 50_000
NATIVE_NUMERIC_PROFILE_MIN_CELLS = 1_000_000


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def series_to_dict(series):
    return {str(k): clean_value(v) for k, v in series.to_dict().items()}


def detect_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    date_cols = []

    for col in df.columns:
        if col in numeric_cols:
            continue
        sample = df[col].dropna().astype(str).head(25)
        if len(sample) == 0:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.7:
            date_cols.append(col)

    return numeric_cols, categorical_cols, date_cols


def _missing_counts(
    df: pd.DataFrame,
    *,
    precomputed: dict[str, int] | None = None,
) -> pd.Series:
    """Count missing values without materializing a frame-sized boolean table."""
    known = precomputed or {}
    return pd.Series(
        {
            column: int(known[column]) if column in known else int(df[column].isna().sum())
            for column in df.columns
        },
        dtype="int64",
    )


def _duplicate_profile(df: pd.DataFrame) -> tuple[int, dict]:
    """Return duplicate estimate plus explicit execution metadata."""
    rows, columns = df.shape
    cells = rows * columns
    if cells <= MAX_EXACT_DUPLICATE_CELLS:
        count = int(df.duplicated().sum())
        return count, {
            "method": "exact",
            "sampled": False,
            "sample_rows": rows,
        }

    sample_rows = min(rows, DUPLICATE_SAMPLE_ROWS)
    positions = _deterministic_stratified_positions(rows, sample_rows)
    sample = df.iloc[positions]
    sample_duplicates = int(sample.duplicated().sum())
    rate = sample_duplicates / max(len(sample), 1)
    estimate = int(round(rate * rows))
    return estimate, {
        "method": "sample_estimate",
        "sampled": True,
        "sample_rows": int(len(sample)),
        "source_rows": int(rows),
        "estimated_duplicate_rate": round(float(rate), 6),
        "strategy": "stratified_jitter_global_rows",
    }


def _bounded_correlations(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[dict, dict]:
    if len(numeric_cols) < 2:
        return {}, {
            "method": "not_applicable",
            "columns_used": len(numeric_cols),
            "columns_available": len(numeric_cols),
            "truncated": False,
        }

    selected = numeric_cols
    truncated = len(selected) > MAX_CORRELATION_COLUMNS
    if truncated:
        non_missing = {
            column: int(df[column].notna().sum())
            for column in numeric_cols
        }
        selected = sorted(
            numeric_cols,
            key=lambda column: (-non_missing[column], str(column)),
        )[:MAX_CORRELATION_COLUMNS]

    finite_numeric = df[selected].replace([np.inf, -np.inf], np.nan)
    correlations = (
        finite_numeric
        .corr(numeric_only=True)
        .round(3)
        .replace({np.nan: None})
        .to_dict()
    )
    return correlations, {
        "method": "dense_bounded" if truncated else "dense_exact_columns",
        "columns_used": len(selected),
        "columns_available": len(numeric_cols),
        "truncated": truncated,
        "max_columns": MAX_CORRELATION_COLUMNS,
    }


def _round_optional(value, digits: int = 3):
    if value is None:
        return None
    return round(float(value), digits)


def _pandas_numeric_summary(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[dict, dict, dict[str, int]]:
    if not numeric_cols:
        return {}, {
            "backend": "not_applicable",
            "method": "not_applicable",
            "approximate_quantiles": False,
        }, {}

    finite_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    summary = (
        finite_numeric
        .describe()
        .T
        .replace({np.nan: None})
        .round(3)
        .to_dict(orient="index")
    )
    return summary, {
        "backend": "pandas",
        "method": "pandas_describe",
        "approximate_quantiles": False,
        "finite_only_moments": True,
        "columns_profiled": len(numeric_cols),
    }, {}


def _native_numeric_summary(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[dict, dict, dict[str, int]]:
    summary: dict[str, dict] = {}
    missing: dict[str, int] = {}
    relative_accuracy = None

    for stream_id, column in enumerate(numeric_cols):
        payload = numeric_profile(df[column], backend="rust", stream_id=stream_id)
        quantiles = payload.get("quantiles", {})
        relative_accuracy = quantiles.get("relative_accuracy", relative_accuracy)
        missing[column] = int(payload["missing"])
        summary[column] = {
            "count": int(payload["count"]),
            "mean": _round_optional(payload.get("mean")),
            "std": _round_optional(payload.get("std")),
            "min": _round_optional(payload.get("minimum")),
            "25%": _round_optional(quantiles.get("p25")),
            "50%": _round_optional(quantiles.get("p50")),
            "75%": _round_optional(quantiles.get("p75")),
            "max": _round_optional(payload.get("maximum")),
        }

    return summary, {
        "backend": "rust",
        "method": "native_fused_numeric_column_scan",
        "approximate_quantiles": True,
        "quantile_relative_accuracy": relative_accuracy,
        "finite_only_moments": True,
        "columns_profiled": len(numeric_cols),
        "raw_observations_retained": False,
    }, missing


def _numeric_summary(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[dict, dict, dict[str, int]]:
    if not numeric_cols:
        return _pandas_numeric_summary(df, numeric_cols)

    requested = os.getenv("FRAMEVITALS_BACKEND", "auto").strip().lower()
    selected = resolve_numeric_backend()
    numeric_cells = int(len(df)) * len(numeric_cols)
    native_worthwhile = (
        len(df) >= NATIVE_NUMERIC_PROFILE_MIN_ROWS
        or numeric_cells >= NATIVE_NUMERIC_PROFILE_MIN_CELLS
    )
    use_native = selected == "rust" and (requested == "rust" or native_worthwhile)

    if not use_native:
        summary, metadata, missing = _pandas_numeric_summary(df, numeric_cols)
        metadata["native_eligible"] = selected == "rust"
        metadata["native_threshold_reached"] = native_worthwhile
        return summary, metadata, missing

    try:
        return _native_numeric_summary(df, numeric_cols)
    except Exception as exc:
        if requested == "rust":
            raise
        summary, metadata, missing = _pandas_numeric_summary(df, numeric_cols)
        metadata.update({
            "native_eligible": True,
            "native_threshold_reached": True,
            "fallback_from": "rust",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        })
        return summary, metadata, missing


def build_profile(df):
    rows, columns = df.shape
    numeric_cols, categorical_cols, date_cols = detect_column_types(df)

    numeric_summary, numeric_summary_metadata, numeric_missing = _numeric_summary(
        df,
        numeric_cols,
    )
    missing_counts = _missing_counts(df, precomputed=numeric_missing)
    missing_percent = (missing_counts / max(rows, 1) * 100).round(2)
    duplicate_rows, duplicate_metadata = _duplicate_profile(df)

    categorical_summary = {}
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False).head(10)
        categorical_summary[col] = {
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in counts.items()},
        }

    correlations, correlation_metadata = _bounded_correlations(df, numeric_cols)

    preview_frame = df.head(15)
    preview = preview_frame.where(preview_frame.notna(), None).to_dict(orient="records")

    return {
        "shape": {"rows": rows, "columns": columns},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "missing_counts": series_to_dict(missing_counts),
        "missing_percent": series_to_dict(missing_percent),
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": round(duplicate_rows / max(rows, 1) * 100, 2),
        "duplicate_metadata": duplicate_metadata,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "numeric_summary": numeric_summary,
        "numeric_summary_metadata": numeric_summary_metadata,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "correlation_metadata": correlation_metadata,
        "preview": preview,
    }
