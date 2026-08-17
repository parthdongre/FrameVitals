"""
Explainability (WS-4)
=====================
SHAP-first global + per-row explanations for the leaderboard winner.

Strategy:
    1. Re-fit the winning estimator with the shared preprocessor.
    2. Try shap.TreeExplainer for tree-based models (RF, GB, XGB, LGBM).
       For linear models, use shap.LinearExplainer.
       Fallback: sklearn permutation importance (model-agnostic).
    3. Collapse one-hot expansions back to original feature names.
    4. Save a beeswarm summary plot when plotting support is installed.

Public entry point:
    explain_winner(df, target_column, leaderboard_result) -> dict
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline

from framevitals.ml_preprocessing import (
    build_sklearn_preprocessor,
    get_transformed_feature_names,
    prepare_ml_matrix,
)


CHART_DIR = Path("static/charts")


# ---------------------------------------------------------------------------
# Estimator rebuilds (mirrors model_leaderboard registry)
# ---------------------------------------------------------------------------
def _rebuild_estimator(name: str, task_type: str):
    """Recreate the estimator by name. Returns None if unknown/missing."""
    if task_type == "classification":
        if name == "DummyClassifier":
            return DummyClassifier(strategy="most_frequent")
        if name == "LogisticRegression":
            return LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            )
        if name == "KNeighborsClassifier":
            return KNeighborsClassifier(n_neighbors=7)
        if name == "RandomForestClassifier":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        if name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                random_state=42,
            )
        if name == "XGBClassifier":
            try:
                from xgboost import XGBClassifier

                return XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    verbosity=0,
                    eval_metric="mlogloss",
                )
            except Exception:
                return None
        if name == "LGBMClassifier":
            try:
                from lightgbm import LGBMClassifier

                return LGBMClassifier(
                    n_estimators=200,
                    max_depth=-1,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            except Exception:
                return None
    else:
        if name == "DummyRegressor":
            return DummyRegressor(strategy="mean")
        if name == "Ridge":
            return Ridge(alpha=1.0, random_state=42)
        if name == "Lasso":
            return Lasso(alpha=0.01, max_iter=20000, random_state=42)
        if name == "KNeighborsRegressor":
            return KNeighborsRegressor(n_neighbors=7)
        if name == "RandomForestRegressor":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
        if name == "GradientBoostingRegressor":
            return GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                random_state=42,
            )
        if name == "XGBRegressor":
            try:
                from xgboost import XGBRegressor

                return XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    verbosity=0,
                )
            except Exception:
                return None
        if name == "LGBMRegressor":
            try:
                from lightgbm import LGBMRegressor

                return LGBMRegressor(
                    n_estimators=200,
                    max_depth=-1,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            except Exception:
                return None

    return None


def _is_tree_model(name: str) -> bool:
    return any(
        token in name
        for token in (
            "RandomForest",
            "GradientBoosting",
            "XGB",
            "LGBM",
        )
    )


def _is_linear_model(name: str) -> bool:
    return name in {"LogisticRegression", "Ridge", "Lasso", "ElasticNet"}


# ---------------------------------------------------------------------------
# One-hot collapse
# ---------------------------------------------------------------------------
def _collapse_to_original(
    feature_names: list[str],
    importances: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[dict]:
    """Sum one-hot expansions back to their original column names."""
    collapsed: dict[str, float] = {
        col: 0.0 for col in numeric_features + categorical_features
    }

    for fname, value in zip(feature_names, importances):
        if fname in collapsed:
            collapsed[fname] += float(value)
            continue
        matched = False
        for col in categorical_features:
            if fname.startswith(col + "_"):
                collapsed[col] = collapsed.get(col, 0.0) + float(value)
                matched = True
                break
        if not matched:
            collapsed[fname] = float(value)

    rows = [
        {"feature": key, "importance": round(value, 6)}
        for key, value in collapsed.items()
    ]
    rows.sort(key=lambda row: abs(row["importance"]), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Optional SHAP plot
# ---------------------------------------------------------------------------
def _save_summary_plot(
    shap_values,
    X_transformed,
    feature_names,
    dataset_id: str,
) -> str | None:
    """Write a SHAP summary plot when optional plotting support is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except Exception:
        return None

    try:
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values,
            features=X_transformed,
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        path = CHART_DIR / f"{dataset_id}_shap_summary.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        return str(path)
    except Exception:
        return None
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Permutation importance fallback
# ---------------------------------------------------------------------------
def _permutation_importance_block(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(
                pipeline,
                X,
                y,
                n_repeats=5,
                random_state=42,
                n_jobs=-1,
                scoring=None,
            )
        rows = [
            {
                "feature": col,
                "importance": round(float(result.importances_mean[index]), 6),
                "std": round(float(result.importances_std[index]), 6),
            }
            for index, col in enumerate(X.columns)
        ]
        rows.sort(key=lambda row: abs(row["importance"]), reverse=True)
        return {"available": True, "method": "permutation", "top_features": rows[:15]}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def explain_winner(
    df: pd.DataFrame,
    target_column: str,
    leaderboard_result: dict,
    dataset_id: str = "explain",
    sample_size: int = 200,
) -> dict:
    """
    Generate SHAP-based global + per-row explanations for the winning model.

    Plot generation is optional. The structured explanation and permutation
    fallback remain usable without Matplotlib/Seaborn.
    """
    if not leaderboard_result.get("available") or not leaderboard_result.get("winner"):
        return {"available": False, "message": "No leaderboard winner to explain."}

    winner_name = leaderboard_result["winner"]["model"]
    task_type = leaderboard_result["task_type"]

    estimator = _rebuild_estimator(winner_name, task_type)
    if estimator is None:
        return {
            "available": False,
            "message": f"Could not rebuild estimator: {winner_name}",
        }

    prep = prepare_ml_matrix(df, target=target_column)
    if not prep["usable"]:
        return {
            "available": False,
            "message": "Preprocessing produced no usable features.",
        }

    X = prep["X"]
    y = prep["y"]
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes

    numeric_features = prep["numeric_features"]
    categorical_features = prep["categorical_features"]

    preprocessor = build_sklearn_preprocessor(
        numeric_features,
        categorical_features,
    )
    pipeline = Pipeline([("pre", preprocessor), ("model", estimator)])

    try:
        stratify = (
            y
            if (
                task_type == "classification"
                and y.nunique() > 1
                and y.value_counts().min() >= 2
            )
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=stratify,
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )

    pipeline.fit(X_train, y_train)

    if len(X_test) > sample_size:
        X_test_sample = X_test.sample(sample_size, random_state=42)
    else:
        X_test_sample = X_test

    fitted_pre = pipeline.named_steps["pre"]
    fitted_model = pipeline.named_steps["model"]
    X_test_transformed = fitted_pre.transform(X_test_sample)
    feature_names = get_transformed_feature_names(
        fitted_pre,
        numeric_features,
        categorical_features,
    )

    method = None
    summary_chart_path: str | None = None
    global_rows: list[dict] = []
    per_row_stories: list[dict] = []
    error_messages: list[str] = []

    try:
        import shap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if _is_tree_model(winner_name):
                explainer = shap.TreeExplainer(fitted_model)
                shap_values_raw = explainer.shap_values(X_test_transformed)
                method = "shap.TreeExplainer"
            elif _is_linear_model(winner_name):
                background = fitted_pre.transform(
                    X_train.sample(min(100, len(X_train)), random_state=42)
                )
                explainer = shap.LinearExplainer(fitted_model, background)
                shap_values_raw = explainer.shap_values(X_test_transformed)
                method = "shap.LinearExplainer"
            else:
                shap_values_raw = None

        if shap_values_raw is not None:
            if isinstance(shap_values_raw, list):
                shap_values = np.mean(
                    [np.abs(array) for array in shap_values_raw],
                    axis=0,
                )
                shap_values_for_plot = (
                    shap_values_raw[1]
                    if len(shap_values_raw) == 2
                    else shap_values_raw[0]
                )
            elif shap_values_raw.ndim == 3:
                shap_values = np.mean(np.abs(shap_values_raw), axis=2)
                shap_values_for_plot = shap_values_raw[:, :, 0]
            else:
                shap_values = np.abs(shap_values_raw)
                shap_values_for_plot = shap_values_raw

            mean_abs = shap_values.mean(axis=0)
            global_rows = _collapse_to_original(
                feature_names,
                mean_abs,
                numeric_features,
                categorical_features,
            )

            row_totals = shap_values.sum(axis=1)
            top_idx = np.argsort(-row_totals)[:3]
            for rank, row_index in enumerate(top_idx, start=1):
                contributions = list(
                    zip(feature_names, shap_values_for_plot[row_index])
                )
                contributions.sort(key=lambda item: abs(item[1]), reverse=True)
                story = [
                    {
                        "feature": feature,
                        "shap_value": round(float(value), 6),
                    }
                    for feature, value in contributions[:5]
                ]
                source_index = X_test_sample.index[row_index]
                per_row_stories.append({
                    "rank": rank,
                    "row_index": (
                        int(source_index)
                        if hasattr(source_index, "__int__")
                        else str(source_index)
                    ),
                    "top_contributions": story,
                })

            summary_chart_path = _save_summary_plot(
                shap_values_for_plot,
                X_test_transformed,
                feature_names,
                dataset_id,
            )

    except Exception as exc:
        error_messages.append(f"SHAP failed: {exc}")
        method = None

    perm = _permutation_importance_block(
        pipeline,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
    )

    if not global_rows and perm.get("available"):
        global_rows = perm["top_features"]
        method = method or "permutation_importance"

    if method is None:
        return {
            "available": False,
            "message": "Both SHAP and permutation importance failed.",
            "errors": error_messages,
        }

    return {
        "available": True,
        "model": winner_name,
        "task_type": task_type,
        "method": method,
        "global_importance": global_rows[:15],
        "per_row_stories": per_row_stories,
        "permutation_importance": perm,
        "summary_chart_path": summary_chart_path,
        "n_test_rows_explained": int(len(X_test_sample)),
        "errors": error_messages,
    }
