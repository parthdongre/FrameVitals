from __future__ import annotations

import numpy as np
import pandas as pd

MAX_CORRELATION_COLUMNS = 100
MAX_EXACT_DUPLICATE_CELLS = 50_000_000
DUPLICATE_SAMPLE_ROWS = 50_000


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


def _missing_counts(df: pd.DataFrame) -> pd.Series:
    """Count missing values without materializing a frame-sized boolean table."""
    return pd.Series(
        {column: int(df[column].isna().sum()) for column in df.columns},
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
    positions = np.linspace(0, rows - 1, num=sample_rows, dtype=np.int64)
    sample = df.iloc[np.unique(positions)]
    sample_duplicates = int(sample.duplicated().sum())
    rate = sample_duplicates / max(len(sample), 1)
    estimate = int(round(rate * rows))
    return estimate, {
        "method": "sample_estimate",
        "sampled": True,
        "sample_rows": int(len(sample)),
        "source_rows": int(rows),
        "estimated_duplicate_rate": round(float(rate), 6),
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
        # Prefer columns with the most usable observations. This is a safe
        # compatibility fallback; the sparse relationship graph will replace
        # dense truncation for ultra-wide datasets.
        non_missing = {
            column: int(df[column].notna().sum())
            for column in numeric_cols
        }
        selected = sorted(
            numeric_cols,
            key=lambda column: (-non_missing[column], str(column)),
        )[:MAX_CORRELATION_COLUMNS]

    correlations = (
        df[selected]
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


def build_profile(df):
    rows, columns = df.shape
    numeric_cols, categorical_cols, date_cols = detect_column_types(df)

    missing_counts = _missing_counts(df)
    missing_percent = (missing_counts / max(rows, 1) * 100).round(2)
    duplicate_rows, duplicate_metadata = _duplicate_profile(df)

    numeric_summary = {}
    if numeric_cols:
        numeric_summary = (
            df[numeric_cols]
            .describe()
            .T
            .replace({np.nan: None})
            .round(3)
            .to_dict(orient="index")
        )

    categorical_summary = {}
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=False).head(10)
        categorical_summary[col] = {
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in counts.items()},
        }

    correlations, correlation_metadata = _bounded_correlations(df, numeric_cols)

    preview = df.head(15).where(df.head(15).notna(), None).to_dict(orient="records")

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
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "correlation_metadata": correlation_metadata,
        "preview": preview,
    }
