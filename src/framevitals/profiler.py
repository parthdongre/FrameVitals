import pandas as pd
import numpy as np

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
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    date_cols = []

    for col in df.columns:
        if col in numeric_cols:
            continue
        sample = df[col].dropna().astype(str).head(25)
        if len(sample) == 0:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            date_cols.append(col)

    return numeric_cols, categorical_cols, date_cols

def build_profile(df):
    rows, columns = df.shape
    numeric_cols, categorical_cols, date_cols = detect_column_types(df)

    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / max(rows, 1) * 100).round(2)
    duplicate_rows = int(df.duplicated().sum())

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

    correlations = {}
    if len(numeric_cols) >= 2:
        correlations = (
            df[numeric_cols]
            .corr(numeric_only=True)
            .round(3)
            .replace({np.nan: None})
            .to_dict()
        )

    preview = df.head(15).where(df.notna(), None).to_dict(orient="records")

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
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "preview": preview,
    }
