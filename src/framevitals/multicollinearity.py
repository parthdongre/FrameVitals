import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


def prepare_numeric_matrix(df, target_column=None, max_columns=30):
    numeric = df.select_dtypes(include=[np.number]).copy()

    if target_column in numeric.columns:
        numeric = numeric.drop(columns=[target_column])

    # Drop constant columns.
    for column in list(numeric.columns):
        if numeric[column].nunique(dropna=True) <= 1:
            numeric = numeric.drop(columns=[column])

    if numeric.shape[1] > max_columns:
        # Keep columns with highest non-missing counts.
        keep = numeric.notna().sum().sort_values(ascending=False).head(max_columns).index.tolist()
        numeric = numeric[keep]

    return numeric


def calculate_vif_scores(df, target_column=None):
    numeric = prepare_numeric_matrix(df, target_column=target_column)

    if numeric.shape[1] < 2:
        return {
            "available": False,
            "message": "At least two numeric feature columns are required for VIF analysis.",
        }

    imputer = SimpleImputer(strategy="median")
    X_all = pd.DataFrame(
        imputer.fit_transform(numeric),
        columns=numeric.columns,
    )

    results = []

    for column in X_all.columns:
        y = X_all[column]
        X = X_all.drop(columns=[column])

        try:
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)

            if r2 >= 0.999:
                vif = 999.0
            else:
                vif = 1.0 / max(1.0 - r2, 1e-9)

            if vif >= 10:
                severity = "High"
            elif vif >= 5:
                severity = "Medium"
            else:
                severity = "Low"

            results.append(
                {
                    "feature": column,
                    "vif": round(float(vif), 4),
                    "r2_explained_by_other_features": round(float(r2), 4),
                    "severity": severity,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "feature": column,
                    "vif": None,
                    "severity": "Error",
                    "message": str(exc),
                }
            )

    results = sorted(
        results,
        key=lambda item: item["vif"] if item["vif"] is not None else -1,
        reverse=True,
    )

    high_count = sum(1 for item in results if item["severity"] == "High")
    medium_count = sum(1 for item in results if item["severity"] == "Medium")

    if high_count:
        status = "High multicollinearity detected"
    elif medium_count:
        status = "Moderate multicollinearity detected"
    else:
        status = "No major multicollinearity detected"

    return {
        "available": True,
        "status": status,
        "feature_count": len(results),
        "high_vif_count": high_count,
        "medium_vif_count": medium_count,
        "vif_scores": results,
        "interpretation": (
            "VIF estimates whether a feature can be explained by other numeric features. "
            "A VIF above 5 suggests moderate multicollinearity; above 10 suggests high multicollinearity."
        ),
    }


def detect_redundant_feature_groups(df, target_column=None, threshold=0.95):
    numeric = prepare_numeric_matrix(df, target_column=target_column)

    if numeric.shape[1] < 2:
        return {
            "available": False,
            "message": "At least two numeric feature columns are required.",
        }

    corr = numeric.corr(numeric_only=True).abs()

    visited = set()
    groups = []

    for column in corr.columns:
        if column in visited:
            continue

        group = [column]

        for other in corr.columns:
            if other == column or other in visited:
                continue

            value = corr.loc[column, other]
            if pd.notna(value) and value >= threshold:
                group.append(other)

        if len(group) > 1:
            for item in group:
                visited.add(item)

            groups.append(
                {
                    "columns": group,
                    "reason": f"Columns have absolute correlation >= {threshold}.",
                    "recommended_action": "Keep one representative feature and review the others for redundancy.",
                }
            )

    return {
        "available": True,
        "group_count": len(groups),
        "groups": groups,
    }


def run_multicollinearity_analysis(df, target_column=None):
    return {
        "vif": calculate_vif_scores(df, target_column=target_column),
        "redundant_groups": detect_redundant_feature_groups(df, target_column=target_column),
    }
