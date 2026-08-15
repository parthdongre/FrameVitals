import pandas as pd


def detect_task_type(series):
    non_missing = series.dropna()
    unique_count = non_missing.nunique()

    if non_missing.empty:
        return {
            "task_type": "unknown",
            "reason": "Target column has no non-missing values.",
        }

    if pd.api.types.is_numeric_dtype(series):
        unique_ratio = unique_count / max(len(non_missing), 1)

        if unique_count <= 10:
            return {
                "task_type": "classification",
                "reason": "Numeric target has low cardinality, so it is treated as classification.",
            }

        if unique_ratio > 0.05:
            return {
                "task_type": "regression",
                "reason": "Numeric target has many unique values, so it is treated as regression.",
            }

        return {
            "task_type": "classification",
            "reason": "Numeric target has limited unique values.",
        }

    return {
        "task_type": "classification",
        "reason": "Non-numeric target is treated as classification.",
    }


def analyze_classification_target(series):
    counts = series.value_counts(dropna=False)
    total = max(len(series), 1)

    classes = []
    for value, count in counts.items():
        classes.append({
            "class": str(value),
            "count": int(count),
            "percent": round(float(count / total * 100), 2),
        })

    non_missing_counts = series.dropna().value_counts()
    if len(non_missing_counts) > 0:
        majority_ratio = round(float(non_missing_counts.iloc[0] / non_missing_counts.sum() * 100), 2)
        minority_count = int(non_missing_counts.iloc[-1])
    else:
        majority_ratio = None
        minority_count = 0

    if majority_ratio is None:
        imbalance_status = "Unknown"
    elif majority_ratio >= 90:
        imbalance_status = "Severely Imbalanced"
    elif majority_ratio >= 75:
        imbalance_status = "Imbalanced"
    elif majority_ratio >= 60:
        imbalance_status = "Slightly Imbalanced"
    else:
        imbalance_status = "Balanced"

    return {
        "class_count": int(series.nunique(dropna=True)),
        "classes": classes,
        "majority_ratio": majority_ratio,
        "minority_class_count": minority_count,
        "imbalance_status": imbalance_status,
    }


def analyze_regression_target(series):
    clean = series.dropna()

    if clean.empty:
        return {
            "count": 0,
            "message": "No non-missing numeric target values.",
        }

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0 or pd.isna(iqr):
        outlier_count = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((clean < lower) | (clean > upper)).sum())

    skewness = clean.skew()

    if abs(skewness) < 0.5:
        skew_label = "Approximately Symmetric"
    elif abs(skewness) < 1:
        skew_label = "Moderately Skewed"
    else:
        skew_label = "Highly Skewed"

    return {
        "count": int(clean.count()),
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "skewness": round(float(skewness), 4),
        "skew_label": skew_label,
        "outlier_count": outlier_count,
    }


def analyze_target(df, target_column):
    if not target_column:
        return {
            "available": False,
            "message": "No target column selected.",
        }

    if target_column not in df.columns:
        return {
            "available": False,
            "message": f"Target column '{target_column}' was not found in the dataset.",
        }

    series = df[target_column]
    missing_count = int(series.isna().sum())
    missing_percent = round(float(series.isna().mean() * 100), 2)
    unique_count = int(series.nunique(dropna=True))

    task = detect_task_type(series)
    task_type = task["task_type"]

    if task_type == "classification":
        details = analyze_classification_target(series)
    elif task_type == "regression":
        details = analyze_regression_target(series)
    else:
        details = {}

    warnings = []

    if missing_percent > 0:
        warnings.append(f"Target column has {missing_percent}% missing values.")

    if task_type == "classification":
        if details.get("imbalance_status") in {"Imbalanced", "Severely Imbalanced"}:
            warnings.append(f"Target appears {details['imbalance_status'].lower()}.")

        if details.get("minority_class_count", 0) < 10:
            warnings.append("Minority class has fewer than 10 samples, so modelling may be unreliable.")

    if task_type == "regression":
        if details.get("skew_label") == "Highly Skewed":
            warnings.append("Regression target is highly skewed; transformation may help.")

        if details.get("outlier_count", 0) > 0:
            warnings.append(f"Regression target has {details['outlier_count']} possible outliers.")

    return {
        "available": True,
        "target_column": target_column,
        "dtype": str(series.dtype),
        "task_type": task_type,
        "task_reason": task["reason"],
        "missing_count": missing_count,
        "missing_percent": missing_percent,
        "unique_count": unique_count,
        "details": details,
        "warnings": warnings,
    }
