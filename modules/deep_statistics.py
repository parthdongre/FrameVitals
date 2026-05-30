import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def safe_float(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None

        if isinstance(value, (np.integer,)):
            return int(value)

        if isinstance(value, (np.floating, float)):
            if math.isinf(float(value)) or math.isnan(float(value)):
                return None
            return round(float(value), 4)

        return value

    except Exception:
        return None


def classify_skewness(skew_value):
    if skew_value is None:
        return "Unknown"

    abs_value = abs(skew_value)

    if abs_value < 0.5:
        return "Approximately Symmetric"

    if abs_value < 1:
        return "Moderately Skewed"

    return "Highly Skewed"


def classify_kurtosis(kurtosis_value):
    if kurtosis_value is None:
        return "Unknown"

    if kurtosis_value > 1:
        return "Heavy-tailed"

    if kurtosis_value < -1:
        return "Light-tailed"

    return "Approximately Normal-tailed"


def normality_test(series):
    clean = series.dropna()

    if len(clean) < 8:
        return {
            "test": "Not enough data",
            "p_value": None,
            "is_probably_normal": None,
            "interpretation": "At least 8 non-missing values are needed.",
        }

    if len(clean) > 5000:
        sample = clean.sample(5000, random_state=42)
    else:
        sample = clean

    try:
        statistic, p_value = stats.normaltest(sample)

        return {
            "test": "D'Agostino-Pearson",
            "p_value": safe_float(p_value),
            "is_probably_normal": bool(p_value >= 0.05),
            "interpretation": (
                "Distribution does not strongly reject normality."
                if p_value >= 0.05
                else "Distribution likely differs from normal."
            ),
        }

    except Exception as exc:
        return {
            "test": "Failed",
            "p_value": None,
            "is_probably_normal": None,
            "interpretation": str(exc),
        }


def numeric_deep_statistics(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            results[column] = {
                "status": "empty",
                "message": "No numeric values available.",
            }
            continue

        skew_value = safe_float(series.skew())
        kurtosis_value = safe_float(series.kurtosis())

        q1 = safe_float(series.quantile(0.25))
        q3 = safe_float(series.quantile(0.75))
        iqr = safe_float(q3 - q1) if q1 is not None and q3 is not None else None

        z_outlier_count = 0
        if len(series) > 1 and series.std() != 0:
            z_scores = np.abs(stats.zscore(series, nan_policy="omit"))
            z_outlier_count = int((z_scores > 3).sum())

        results[column] = {
            "count": int(series.count()),
            "mean": safe_float(series.mean()),
            "median": safe_float(series.median()),
            "std": safe_float(series.std()),
            "min": safe_float(series.min()),
            "max": safe_float(series.max()),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "skewness": skew_value,
            "skewness_label": classify_skewness(skew_value),
            "kurtosis": kurtosis_value,
            "kurtosis_label": classify_kurtosis(kurtosis_value),
            "z_score_outliers": z_outlier_count,
            "normality": normality_test(series),
        }

    return results


def correlation_insights(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_columns) < 2:
        return {
            "available": False,
            "message": "Not enough numeric columns for correlation analysis.",
            "top_pairs": [],
        }

    corr = df[numeric_columns].corr(numeric_only=True)

    pairs = []

    for i, col_a in enumerate(numeric_columns):
        for col_b in numeric_columns[i + 1:]:
            value = corr.loc[col_a, col_b]

            if pd.isna(value):
                continue

            pairs.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": safe_float(value),
                    "strength": classify_correlation(value),
                }
            )

    pairs = sorted(pairs, key=lambda item: abs(item["correlation"]), reverse=True)

    return {
        "available": True,
        "top_pairs": pairs[:10],
        "high_correlation_pairs": [
            pair for pair in pairs if abs(pair["correlation"]) >= 0.8
        ],
    }


def classify_correlation(value):
    abs_value = abs(value)

    if abs_value >= 0.8:
        return "Very Strong"

    if abs_value >= 0.6:
        return "Strong"

    if abs_value >= 0.4:
        return "Moderate"

    if abs_value >= 0.2:
        return "Weak"

    return "Very Weak"


def categorical_deep_statistics(df):
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    results = {}

    for column in categorical_columns:
        counts = df[column].value_counts(dropna=False)
        total = max(len(df), 1)

        top_values = []

        for value, count in counts.head(10).items():
            top_values.append(
                {
                    "value": str(value),
                    "count": int(count),
                    "percent": round(float(count / total * 100), 2),
                }
            )

        unique_count = int(df[column].nunique(dropna=True))

        if unique_count <= 1:
            cardinality_label = "Constant"
        elif unique_count <= 10:
            cardinality_label = "Low Cardinality"
        elif unique_count <= 50:
            cardinality_label = "Medium Cardinality"
        else:
            cardinality_label = "High Cardinality"

        results[column] = {
            "unique_count": unique_count,
            "cardinality_label": cardinality_label,
            "top_values": top_values,
        }

    return results


def chi_square_relationships(df):
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    results = []

    if len(categorical_columns) < 2:
        return results

    for i, col_a in enumerate(categorical_columns):
        for col_b in categorical_columns[i + 1:]:
            try:
                table = pd.crosstab(df[col_a], df[col_b])

                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                chi2, p_value, dof, expected = stats.chi2_contingency(table)

                results.append(
                    {
                        "column_a": col_a,
                        "column_b": col_b,
                        "chi2": safe_float(chi2),
                        "p_value": safe_float(p_value),
                        "degrees_of_freedom": int(dof),
                        "relationship_likely": bool(p_value < 0.05),
                        "interpretation": (
                            "Possible relationship detected."
                            if p_value < 0.05
                            else "No strong relationship detected."
                        ),
                    }
                )

            except Exception:
                continue

    return sorted(results, key=lambda item: item["p_value"] if item["p_value"] is not None else 1)


def run_deep_statistics(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        return {
            "numeric_statistics": numeric_deep_statistics(df),
            "categorical_statistics": categorical_deep_statistics(df),
            "correlation_insights": correlation_insights(df),
            "chi_square_relationships": chi_square_relationships(df),
        }