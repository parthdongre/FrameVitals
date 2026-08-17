from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from framevitals.provenance import normalize_execution


def calculate_outlier_percent(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 0.0, {}

    total_cells = len(df) * len(numeric_cols)
    outlier_count = 0
    details = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            details[col] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        outlier_count += count
        details[col] = count

    percent = round(outlier_count / max(total_cells, 1) * 100, 2)
    return percent, details


def _health_label(overall: float) -> str:
    if overall >= 90:
        return "Excellent"
    if overall >= 75:
        return "Good"
    if overall >= 60:
        return "Moderate"
    if overall >= 40:
        return "Poor"
    return "Critical"


def _score_payload(
    *,
    missing_percent: float,
    duplicate_percent: float,
    outlier_percent: float,
    constant_columns: list[str],
    high_cardinality_columns: list[str],
    columns: int,
    outlier_details: Mapping[str, Any],
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    constant_percent = len(constant_columns) / max(columns, 1) * 100
    high_cardinality_percent = len(high_cardinality_columns) / max(columns, 1) * 100

    completeness_score = max(0, round(100 - missing_percent, 2))
    uniqueness_score = max(0, round(100 - duplicate_percent, 2))
    outlier_safety_score = max(0, round(100 - outlier_percent, 2))
    consistency_score = max(
        0,
        round(100 - constant_percent - high_cardinality_percent, 2),
    )

    overall = round(
        completeness_score * 0.30
        + consistency_score * 0.20
        + uniqueness_score * 0.20
        + outlier_safety_score * 0.20
        + 10,
        2,
    )
    overall = max(0, min(100, overall))

    payload: dict[str, Any] = {
        "overall_score": overall,
        "label": _health_label(overall),
        "components": {
            "completeness": completeness_score,
            "consistency": consistency_score,
            "uniqueness": uniqueness_score,
            "outlier_safety": outlier_safety_score,
        },
        "details": {
            "missing_percent": round(missing_percent, 2),
            "duplicate_percent": duplicate_percent,
            "outlier_percent": outlier_percent,
            "constant_columns": constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "outlier_details": dict(outlier_details),
        },
    }
    if execution is not None:
        payload["execution"] = dict(execution)
    return payload


def _profile_constant_and_cardinality_columns(
    profile: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    """Infer consistency signals from a profile without raw full-row access."""
    rows = int(profile.get("shape", {}).get("rows", 0) or 0)
    numeric_summary = profile.get("numeric_summary", {})
    categorical_summary = profile.get("categorical_summary", {})

    constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []

    if isinstance(numeric_summary, Mapping):
        for column, raw_summary in numeric_summary.items():
            if not isinstance(raw_summary, Mapping):
                continue
            count = int(raw_summary.get("count", 0) or 0)
            minimum = raw_summary.get("min")
            maximum = raw_summary.get("max")
            if count > 0 and minimum is not None and maximum is not None and minimum == maximum:
                constant_columns.append(str(column))

    if isinstance(categorical_summary, Mapping):
        for column, raw_summary in categorical_summary.items():
            if not isinstance(raw_summary, Mapping):
                continue
            unique_values = int(raw_summary.get("unique_values", 0) or 0)
            if unique_values <= 1:
                constant_columns.append(str(column))
            if rows > 0 and unique_values > 0.8 * rows:
                high_cardinality_columns.append(str(column))

    return sorted(set(constant_columns)), sorted(set(high_cardinality_columns))


def calculate_health_score_from_profile_sample(
    profile: Mapping[str, Any],
    sample: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate a bounded health score from a full profile plus row sample.

    On ultra-wide sources the profile may cover all rows but a deterministic
    subset of columns. Completeness and consistency are then estimates over that
    projected column sample rather than being diluted by the unprofiled width.
    """
    rows = max(int(profile.get("shape", {}).get("rows", 0) or 0), 1)
    source_columns = max(int(profile.get("shape", {}).get("columns", 0) or 0), 1)
    streaming_metadata = profile.get("streaming_metadata", {})
    profiled_columns = source_columns
    if isinstance(streaming_metadata, Mapping):
        profiled_columns = max(
            int(streaming_metadata.get("profiled_columns", source_columns) or source_columns),
            1,
        )
    column_sampled = profiled_columns < source_columns

    missing_counts = profile.get("missing_counts", {})
    missing_total = sum(
        int(value)
        for value in missing_counts.values()
        if isinstance(value, (int, np.integer))
    ) if isinstance(missing_counts, Mapping) else 0
    missing_percent = missing_total / (rows * profiled_columns) * 100
    duplicate_percent = float(profile.get("duplicate_percent", 0.0) or 0.0)

    outlier_percent, sample_outlier_details = calculate_outlier_percent(sample)
    sample_rows = len(sample)
    outlier_scale = rows / max(sample_rows, 1)
    outlier_details = {
        column: int(round(count * outlier_scale))
        for column, count in sample_outlier_details.items()
    }
    constant_columns, high_cardinality_columns = _profile_constant_and_cardinality_columns(
        profile
    )

    duplicate_metadata = profile.get("duplicate_metadata", {})
    duplicate_estimated = bool(
        isinstance(duplicate_metadata, Mapping) and duplicate_metadata.get("sampled")
    )

    execution = normalize_execution(
        {
            "method": "streaming_profile_with_bounded_row_sample",
            "source_rows": rows,
            "source_columns": source_columns,
            "profiled_columns": profiled_columns,
            "column_sampled": column_sampled,
            "sample_rows": sample_rows,
            "sampled": sample_rows < rows or column_sampled,
            "full_materialization": False,
            "components": {
                "completeness": (
                    "full_rows_projected_columns_estimate"
                    if column_sampled
                    else "full_stream_exact"
                ),
                "uniqueness": (
                    "projected_columns_row_sample_estimate"
                    if column_sampled
                    else (
                        "full_stream_sample_estimate"
                        if duplicate_estimated
                        else "full_stream_exact"
                    )
                ),
                "consistency": (
                    "full_rows_projected_columns_estimate"
                    if column_sampled
                    else "full_stream_profile"
                ),
                "outlier_safety": (
                    "projected_columns_bounded_row_sample_estimate"
                    if column_sampled
                    else (
                        "bounded_row_sample_estimate" if sample_rows < rows else "exact"
                    )
                ),
            },
        },
        method="streaming_profile_with_bounded_row_sample",
        full_materialization=False,
    )
    return _score_payload(
        missing_percent=missing_percent,
        duplicate_percent=duplicate_percent,
        outlier_percent=outlier_percent,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        columns=profiled_columns,
        outlier_details=outlier_details,
        execution=execution,
    )


def calculate_health_score(df: pd.DataFrame, profile: Mapping[str, Any]):
    rows = max(int(profile["shape"]["rows"]), 1)
    columns = max(int(profile["shape"]["columns"]), 1)

    missing_total = sum(
        value for value in profile["missing_counts"].values() if isinstance(value, int)
    )
    missing_percent = missing_total / (rows * columns) * 100
    duplicate_percent = profile["duplicate_percent"]
    outlier_percent, outlier_details = calculate_outlier_percent(df)

    constant_columns = []
    high_cardinality_columns = []

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        if unique_count <= 1:
            constant_columns.append(col)
        if df[col].dtype == "object" and unique_count > 0.8 * rows:
            high_cardinality_columns.append(col)

    return _score_payload(
        missing_percent=missing_percent,
        duplicate_percent=duplicate_percent,
        outlier_percent=outlier_percent,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        columns=columns,
        outlier_details=outlier_details,
    )
