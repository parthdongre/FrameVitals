import numpy as np

def calculate_ml_readiness(df):
    rows, columns = df.shape

    missing_percent = float(df.isna().sum().sum() / max(rows * columns, 1) * 100)
    duplicate_percent = float(df.duplicated().sum() / max(rows, 1) * 100)

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

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
