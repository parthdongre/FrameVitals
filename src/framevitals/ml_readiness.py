import numpy as np


_CATEGORICAL_DTYPES = ["object", "string", "category", "bool"]


def calculate_ml_readiness(df, profile=None):
    """Calculate ML-readiness while reusing profile metrics when available.

    ``profile`` is optional to preserve the standalone helper API. The main
    pipeline passes the already-built profile so FrameVitals does not rescan
    the full dataset for missing values, duplicates, and basic column groups.
    """
    rows, columns = df.shape

    if profile is None:
        missing_percent = float(df.isna().sum().sum() / max(rows * columns, 1) * 100)
        duplicate_percent = float(df.duplicated().sum() / max(rows, 1) * 100)
        categorical_cols = df.select_dtypes(include=_CATEGORICAL_DTYPES).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing_total = sum(
            int(value)
            for value in profile.get("missing_counts", {}).values()
            if value is not None
        )
        missing_percent = float(missing_total / max(rows * columns, 1) * 100)
        duplicate_percent = float(profile.get("duplicate_percent", 0.0))
        categorical_cols = list(profile.get("categorical_columns", []))
        numeric_cols = list(profile.get("numeric_columns", []))

    encoding_penalty = min(20, len(categorical_cols) * 2)
    missing_penalty = min(30, missing_percent)
    duplicate_penalty = min(15, duplicate_percent)

    score = 100 - missing_penalty - duplicate_penalty - encoding_penalty
    score = round(max(0, min(100, score)), 2)

    if score >= 85:
        label = "Ready"
    elif score >= 70:
        label = "Mostly Ready"
    elif score >= 50:
        label = "Partially Ready"
    else:
        label = "Not Ready"

    return {
        "score": score,
        "label": label,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "issues": {
            "missing_percent": round(missing_percent, 2),
            "duplicate_percent": round(duplicate_percent, 2),
            "encoding_required_count": len(categorical_cols),
        },
        "recommendations": [
            "Handle missing values before model training.",
            "Encode categorical columns.",
            "Remove duplicates if they are not valid repeated records.",
            "Select a clear target column for supervised learning.",
        ],
    }
