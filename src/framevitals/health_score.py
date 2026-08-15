import pandas as pd
import numpy as np

def calculate_outlier_percent(df):
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

def calculate_health_score(df, profile):
    rows = max(profile["shape"]["rows"], 1)
    columns = max(profile["shape"]["columns"], 1)

    missing_total = sum(
        value for value in profile["missing_counts"].values()
        if isinstance(value, int)
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

    constant_percent = len(constant_columns) / columns * 100
    high_cardinality_percent = len(high_cardinality_columns) / columns * 100

    completeness_score = max(0, round(100 - missing_percent, 2))
    uniqueness_score = max(0, round(100 - duplicate_percent, 2))
    outlier_safety_score = max(0, round(100 - outlier_percent, 2))
    consistency_score = max(0, round(100 - constant_percent - high_cardinality_percent, 2))

    overall = round(
        completeness_score * 0.30
        + consistency_score * 0.20
        + uniqueness_score * 0.20
        + outlier_safety_score * 0.20
        + 10,
        2,
    )

    overall = max(0, min(100, overall))

    if overall >= 90:
        label = "Excellent"
    elif overall >= 75:
        label = "Good"
    elif overall >= 60:
        label = "Moderate"
    elif overall >= 40:
        label = "Poor"
    else:
        label = "Critical"

    return {
        "overall_score": overall,
        "label": label,
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
            "outlier_details": outlier_details,
        },
    }
