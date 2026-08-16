"""Unified target-aware diagnostics for supervised-learning datasets.

This layer connects FrameVitals' existing target and leakage diagnostics and
adds lightweight, interpretable feature-to-target association ranking. It is
not an AutoML trainer and does not fit production models.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from framevitals.column_roles import infer_column_roles
from framevitals.target_analyzer import analyze_target
from framevitals.target_leakage import run_target_leakage_analysis


_EXCLUDED_ROLES = {"id_like", "constant"}


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def _cramers_v(feature: pd.Series, target: pd.Series) -> tuple[float | None, int]:
    mask = feature.notna() & target.notna()
    n = int(mask.sum())
    if n < 10:
        return None, n

    table = pd.crosstab(feature[mask], target[mask])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, n
    if table.shape[0] > 50 or table.shape[1] > 50:
        return None, n

    try:
        chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    except ValueError:
        return None, n

    phi2 = chi2 / n
    rows, cols = table.shape
    denominator = min(rows - 1, cols - 1)
    if denominator <= 0:
        return None, n
    return _safe_float(math.sqrt(max(phi2, 0.0) / denominator)), n


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> tuple[float | None, int]:
    mask = categories.notna() & values.notna()
    n = int(mask.sum())
    if n < 10:
        return None, n

    groups = categories[mask]
    numeric = pd.to_numeric(values[mask], errors="coerce")
    valid = numeric.notna()
    groups = groups[valid]
    numeric = numeric[valid]
    n = len(numeric)
    if n < 10 or groups.nunique(dropna=True) < 2 or groups.nunique(dropna=True) > 50:
        return None, n

    grand_mean = float(numeric.mean())
    total = float(((numeric - grand_mean) ** 2).sum())
    if total <= 0:
        return None, n

    between = 0.0
    for _, group_values in numeric.groupby(groups):
        if group_values.empty:
            continue
        between += len(group_values) * (float(group_values.mean()) - grand_mean) ** 2

    eta_squared = max(0.0, min(1.0, between / total))
    return _safe_float(math.sqrt(eta_squared)), n


def _numeric_association(feature: pd.Series, target: pd.Series) -> tuple[float | None, int, str]:
    mask = feature.notna() & target.notna()
    n = int(mask.sum())
    if n < 10:
        return None, n, "spearman"

    x = pd.to_numeric(feature[mask], errors="coerce")
    y = pd.to_numeric(target[mask], errors="coerce")
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < 10 or x.nunique() < 2 or y.nunique() < 2:
        return None, n, "spearman"

    try:
        coefficient, _ = stats.spearmanr(x, y)
    except Exception:
        return None, n, "spearman"
    return _safe_float(abs(coefficient)), n, "spearman"


def _binary_numeric_association(
    feature: pd.Series,
    target: pd.Series,
) -> tuple[float | None, int, str]:
    mask = feature.notna() & target.notna()
    n = int(mask.sum())
    if n < 10:
        return None, n, "point_biserial"

    classes = list(pd.unique(target[mask]))
    if len(classes) != 2:
        score, n_eta = _correlation_ratio(target, feature)
        return score, n_eta, "correlation_ratio"

    encoded = target[mask].map({classes[0]: 0, classes[1]: 1})
    numeric = pd.to_numeric(feature[mask], errors="coerce")
    valid = numeric.notna() & encoded.notna()
    numeric = numeric[valid]
    encoded = encoded[valid]
    n = len(numeric)
    if n < 10 or numeric.nunique() < 2:
        return None, n, "point_biserial"

    try:
        coefficient, _ = stats.pointbiserialr(encoded.astype(float), numeric.astype(float))
    except Exception:
        return None, n, "point_biserial"
    return _safe_float(abs(coefficient)), n, "point_biserial"


def _association_strength(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.8:
        return "very_strong"
    if score >= 0.6:
        return "strong"
    if score >= 0.4:
        return "moderate"
    if score >= 0.2:
        return "weak"
    return "very_weak"


def _rank_target_associations(
    df: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
    column_roles: dict,
    max_features: int,
) -> list[dict[str, Any]]:
    target = df[target_column]
    target_is_numeric = pd.api.types.is_numeric_dtype(target)
    associations: list[dict[str, Any]] = []

    for column in df.columns:
        if column == target_column:
            continue

        role_info = column_roles.get(column, {})
        role_set = set(role_info.get("roles", []))
        if role_set.intersection(_EXCLUDED_ROLES):
            continue

        feature = df[column]
        feature_is_numeric = pd.api.types.is_numeric_dtype(feature)
        score: float | None = None
        overlap = 0
        method = "unsupported"

        if task_type == "regression" and target_is_numeric:
            if feature_is_numeric:
                score, overlap, method = _numeric_association(feature, target)
            else:
                score, overlap = _correlation_ratio(feature, target)
                method = "correlation_ratio"
        elif task_type == "classification":
            if feature_is_numeric:
                score, overlap, method = _binary_numeric_association(feature, target)
            else:
                score, overlap = _cramers_v(feature, target)
                method = "cramers_v"

        if score is None:
            continue

        associations.append({
            "feature": column,
            "score": score,
            "strength": _association_strength(score),
            "method": method,
            "overlap": int(overlap),
        })

    associations.sort(key=lambda item: (-item["score"], item["feature"]))
    return associations[:max_features]


def _split_guidance(
    target_profile: dict[str, Any],
    column_roles: dict,
) -> dict[str, Any]:
    time_like = [
        column
        for column, info in column_roles.items()
        if "time_like" in info.get("roles", [])
    ]
    task_type = target_profile.get("task_type")
    details = target_profile.get("details", {}) or {}

    if time_like:
        return {
            "strategy": "review_time_aware_split",
            "reason": (
                "Time-like columns are present. If rows represent chronological "
                "events, prefer an ordered/time-based split to avoid future-to-past leakage."
            ),
            "time_candidates": time_like[:10],
        }

    if task_type == "classification" and details.get("class_count", 0) >= 2:
        return {
            "strategy": "stratified_random_split",
            "reason": "Preserve target-class proportions across train/validation/test splits.",
            "time_candidates": [],
        }

    if task_type == "regression":
        return {
            "strategy": "random_split",
            "reason": (
                "No time-like structure was detected; a reproducible random split is a "
                "reasonable baseline for regression."
            ),
            "time_candidates": [],
        }

    return {
        "strategy": "review_manually",
        "reason": "FrameVitals could not infer a safe split strategy from the target metadata.",
        "time_candidates": [],
    }


def run_target_intelligence(
    df: pd.DataFrame,
    *,
    target_column: str | None,
    column_roles: dict | None = None,
    max_features: int = 25,
) -> dict[str, Any]:
    """Run explainable target-quality, leakage, association, and split checks."""
    if not target_column or target_column not in df.columns:
        return {
            "available": False,
            "message": "No valid target column selected.",
        }
    if max_features < 1:
        raise ValueError("max_features must be at least 1.")

    if column_roles is None:
        column_roles = infer_column_roles(df)

    target_profile = analyze_target(df, target_column)
    leakage = run_target_leakage_analysis(df, target_column)
    task_type = target_profile.get("task_type", "unknown")
    associations = _rank_target_associations(
        df,
        target_column=target_column,
        task_type=task_type,
        column_roles=column_roles,
        max_features=max_features,
    )

    target_roles = column_roles.get(target_column, {}).get("roles", [])
    warnings: list[dict[str, str]] = []

    if "id_like" in target_roles:
        warnings.append({
            "code": "target.id_like",
            "severity": "high",
            "message": "Selected target looks identifier-like and may not be a meaningful prediction target.",
        })
    if target_profile.get("missing_percent", 0) >= 20:
        warnings.append({
            "code": "target.high_missingness",
            "severity": "high",
            "message": f"Target has {target_profile['missing_percent']}% missing values.",
        })

    details = target_profile.get("details", {}) or {}
    if task_type == "classification" and details.get("class_count", 0) > 50:
        warnings.append({
            "code": "target.high_cardinality_classification",
            "severity": "medium",
            "message": (
                f"Target has {details['class_count']} classes; confirm this is intentional "
                "before treating the task as classification."
            ),
        })

    for item in leakage.get("warnings", []):
        warnings.append({
            "code": f"target.leakage.{item.get('feature', 'feature')}",
            "severity": str(item.get("risk", "medium")).lower(),
            "message": str(item.get("reason", "Potential target leakage detected.")),
        })

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    warnings.sort(key=lambda item: (severity_order.get(item["severity"], 9), item["code"]))

    return {
        "available": True,
        "target_column": target_column,
        "task_type": task_type,
        "target_profile": target_profile,
        "target_roles": list(target_roles),
        "split_guidance": _split_guidance(target_profile, column_roles),
        "leakage": leakage,
        "top_associations": associations,
        "warning_count": len(warnings),
        "warnings": warnings,
    }
