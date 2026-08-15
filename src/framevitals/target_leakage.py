import pandas as pd
import numpy as np


def same_ratio_non_missing(a, b):
    both_present = a.notna() & b.notna()
    overlap = int(both_present.sum())

    if overlap < 10:
        return None, overlap

    ratio = (
        a.loc[both_present].astype(str).values
        == b.loc[both_present].astype(str).values
    ).mean()

    return round(float(ratio), 4), overlap


def numeric_correlation(a, b):
    both_present = a.notna() & b.notna()
    overlap = int(both_present.sum())

    if overlap < 10:
        return None, overlap

    try:
        corr = a.loc[both_present].corr(b.loc[both_present])
        if pd.isna(corr):
            return None, overlap
        return round(float(corr), 4), overlap
    except Exception:
        return None, overlap


def classify_target_leakage_risk(feature, target, same_ratio, corr):
    lower_feature = feature.lower()
    lower_target = target.lower()

    if same_ratio is not None and same_ratio >= 0.98:
        return "Critical", "Feature values almost exactly match the target on non-missing rows."

    if corr is not None and abs(corr) >= 0.98:
        return "High", "Feature is almost perfectly correlated with the target."

    if corr is not None and abs(corr) >= 0.9:
        return "Medium", "Feature is very strongly correlated with the target."

    if lower_target in lower_feature or lower_feature in lower_target:
        return "Medium", "Feature name is very similar to the target name."

    return "Low", "No strong direct target leakage pattern detected."


def run_target_leakage_analysis(df, target_column):
    if not target_column or target_column not in df.columns:
        return {
            "available": False,
            "message": "No valid target column selected.",
        }

    target = df[target_column]
    warnings = []

    for column in df.columns:
        if column == target_column:
            continue

        feature = df[column]

        same_ratio, same_overlap = same_ratio_non_missing(feature, target)

        corr = None
        corr_overlap = 0
        if pd.api.types.is_numeric_dtype(feature) and pd.api.types.is_numeric_dtype(target):
            corr, corr_overlap = numeric_correlation(feature, target)

        risk, reason = classify_target_leakage_risk(
            feature=column,
            target=target_column,
            same_ratio=same_ratio,
            corr=corr,
        )

        if risk in {"Critical", "High", "Medium"}:
            warnings.append(
                {
                    "feature": column,
                    "target": target_column,
                    "risk": risk,
                    "reason": reason,
                    "same_ratio": same_ratio,
                    "same_overlap": same_overlap,
                    "correlation": corr,
                    "correlation_overlap": corr_overlap,
                }
            )

    risk_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    warnings = sorted(warnings, key=lambda item: risk_order.get(item["risk"], 9))

    if any(item["risk"] == "Critical" for item in warnings):
        status = "Critical Review Required"
    elif any(item["risk"] == "High" for item in warnings):
        status = "High Risk"
    elif warnings:
        status = "Review Recommended"
    else:
        status = "No strong target leakage detected"

    return {
        "available": True,
        "target_column": target_column,
        "status": status,
        "warning_count": len(warnings),
        "warnings": warnings,
        "interpretation": (
            "Target-aware leakage checks compare each feature directly against the selected target. "
            "High or critical warnings can inflate model performance and should be reviewed before trusting ML results."
        ),
    }
