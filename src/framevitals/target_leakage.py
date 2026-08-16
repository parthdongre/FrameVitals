import pandas as pd


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


def categorical_mapping_purity(feature, target, *, max_categories=20):
    """Measure whether low-cardinality feature values determine target labels.

    A value of 1.0 means that every observed feature category maps to exactly one
    target class. High-cardinality columns are intentionally skipped because
    identifiers can trivially memorize a target without representing a useful
    categorical leakage pattern.
    """
    both_present = feature.notna() & target.notna()
    overlap = int(both_present.sum())
    if overlap < 10:
        return None, overlap

    feature_values = feature.loc[both_present]
    target_values = target.loc[both_present]
    feature_cardinality = int(feature_values.nunique(dropna=True))
    target_cardinality = int(target_values.nunique(dropna=True))

    if (
        feature_cardinality < 2
        or feature_cardinality > max_categories
        or target_cardinality < 2
        or target_cardinality > max_categories
    ):
        return None, overlap

    table = pd.crosstab(feature_values, target_values)
    if table.empty:
        return None, overlap

    correctly_determined = int(table.max(axis=1).sum())
    purity = correctly_determined / max(int(table.to_numpy().sum()), 1)
    return round(float(purity), 4), overlap


def classify_target_leakage_risk(
    feature,
    target,
    same_ratio,
    corr,
    mapping_purity=None,
):
    lower_feature = feature.lower()
    lower_target = target.lower()

    if same_ratio is not None and same_ratio >= 0.98:
        return "Critical", "Feature values almost exactly match the target on non-missing rows."

    if mapping_purity is not None and mapping_purity >= 0.995:
        return (
            "Critical",
            "Feature categories almost perfectly determine the target labels.",
        )

    if mapping_purity is not None and mapping_purity >= 0.98:
        return (
            "High",
            "Feature categories very strongly determine the target labels.",
        )

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

        mapping_purity = None
        mapping_overlap = 0
        if not pd.api.types.is_numeric_dtype(feature):
            mapping_purity, mapping_overlap = categorical_mapping_purity(feature, target)

        risk, reason = classify_target_leakage_risk(
            feature=column,
            target=target_column,
            same_ratio=same_ratio,
            corr=corr,
            mapping_purity=mapping_purity,
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
                    "mapping_purity": mapping_purity,
                    "mapping_overlap": mapping_overlap,
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
