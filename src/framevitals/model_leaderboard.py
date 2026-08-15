"""
Model Leaderboard (WS-3)
========================
Cross-validated leaderboard of multiple ML models for a given target.

For classification:
    DummyClassifier, LogisticRegression, KNeighborsClassifier,
    RandomForestClassifier, GradientBoostingClassifier,
    XGBClassifier (optional), LGBMClassifier (optional)

For regression:
    DummyRegressor, Ridge, Lasso, KNeighborsRegressor,
    RandomForestRegressor, GradientBoostingRegressor,
    XGBRegressor (optional), LGBMRegressor (optional)

Each model is wrapped in a Pipeline with the shared preprocessor from
modules.ml_preprocessing.prepare_ml_matrix + build_sklearn_preprocessor.

Validation:
    StratifiedKFold(5) for classification, KFold(5) for regression.
    XGBoost / LightGBM are imported lazily and skipped if missing.

The winner is also fit on the full training data and a calibration check
(classification) or residual summary (regression) is computed on a hold-out.

Public entry point:
    run_model_leaderboard(df, target_column, task_type=None) -> dict
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline

from framevitals.ml_preprocessing import (
    build_sklearn_preprocessor,
    prepare_ml_matrix,
)

# ---------------------------------------------------------------------------
# Optional heavy models — imported lazily so missing libs don't break import
# ---------------------------------------------------------------------------

def _maybe_xgb_classifier():
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


def _maybe_xgb_regressor():
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


def _maybe_lgbm_classifier():
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


def _maybe_lgbm_regressor():
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


# ---------------------------------------------------------------------------
# Task auto-detection
# ---------------------------------------------------------------------------

def _infer_task_type(y: pd.Series) -> str:
    """Best-effort: classification if dtype is non-numeric or low-cardinality."""
    if pd.api.types.is_numeric_dtype(y):
        n = len(y)
        unique = int(y.nunique(dropna=True))
        # Heuristic: integer-like with few unique values is classification
        if unique <= 20 and unique <= max(2, int(n * 0.05)):
            return "classification"
        return "regression"
    return "classification"


# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

def _classification_registry(class_count: int) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, n_jobs=-1, class_weight="balanced", random_state=42
        ),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=7),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, random_state=42
        ),
    }
    xgb = _maybe_xgb_classifier()
    if xgb is not None:
        registry["XGBClassifier"] = xgb
    lgbm = _maybe_lgbm_classifier()
    if lgbm is not None:
        registry["LGBMClassifier"] = lgbm
    return registry


def _regression_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.01, max_iter=20000, random_state=42),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=7),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, random_state=42
        ),
    }
    xgb = _maybe_xgb_regressor()
    if xgb is not None:
        registry["XGBRegressor"] = xgb
    lgbm = _maybe_lgbm_regressor()
    if lgbm is not None:
        registry["LGBMRegressor"] = lgbm
    return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_round(value, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        if not np.isfinite(v):
            return None
        return round(v, ndigits)
    except Exception:
        return None


def _cv_for(task_type: str, y: pd.Series, n_splits: int = 5):
    if task_type == "classification":
        # Reduce splits if smallest class is too tiny
        min_class = int(y.value_counts().min()) if len(y) else 0
        actual_splits = max(2, min(n_splits, min_class))
        return StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42), actual_splits
    return KFold(n_splits=n_splits, shuffle=True, random_state=42), n_splits


def _scoring_for(task_type: str) -> tuple[dict[str, str], str]:
    if task_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
        }
        return scoring, "f1_weighted"
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }
    return scoring, "r2"


def _coerce_classification_target(y: pd.Series) -> pd.Series:
    """XGBoost requires integer-encoded class labels; encode safely."""
    if pd.api.types.is_numeric_dtype(y):
        return y
    return y.astype("category").cat.codes


# ---------------------------------------------------------------------------
# Hold-out diagnostics for the winner
# ---------------------------------------------------------------------------

def _classification_holdout(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict:
    n = len(y)
    test_size = max(50, int(n * 0.25)) if n >= 200 else max(20, int(n * 0.25))
    test_size = min(test_size, n - 20)
    if test_size <= 0:
        return {"available": False, "reason": "not enough rows for holdout"}

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    out: dict[str, Any] = {
        "available": True,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": _safe_round(accuracy_score(y_test, y_pred)),
        "f1_weighted": _safe_round(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "precision_weighted": _safe_round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": _safe_round(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }

    # Brier score for binary calibration
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
            classes = pipeline.classes_ if hasattr(pipeline, "classes_") else None
            if proba.shape[1] == 2 and classes is not None:
                # Binary: pick positive-class column
                pos_idx = 1
                y_bin = (y_test == classes[pos_idx]).astype(int).values
                out["brier_score"] = _safe_round(brier_score_loss(y_bin, proba[:, pos_idx]))
    except Exception as exc:
        out["brier_error"] = str(exc)

    return out


def _regression_holdout(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    n = len(y)
    test_size = max(50, int(n * 0.25)) if n >= 200 else max(20, int(n * 0.25))
    test_size = min(test_size, n - 20)
    if test_size <= 0:
        return {"available": False, "reason": "not enough rows for holdout"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    residuals = y_test.values - y_pred

    return {
        "available": True,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2": _safe_round(r2_score(y_test, y_pred)),
        "mae": _safe_round(mean_absolute_error(y_test, y_pred)),
        "rmse": _safe_round(np.sqrt(mean_squared_error(y_test, y_pred))),
        "residual_summary": {
            "mean": _safe_round(np.mean(residuals)),
            "std": _safe_round(np.std(residuals)),
            "min": _safe_round(np.min(residuals)),
            "max": _safe_round(np.max(residuals)),
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_model_leaderboard(
    df: pd.DataFrame,
    target_column: str,
    task_type: str | None = None,
    n_splits: int = 5,
) -> dict:
    """
    Run a CV-validated model leaderboard.

    Args:
        df: Dataframe.
        target_column: Target column name.
        task_type: "classification" | "regression" | None (auto-detect).
        n_splits: CV folds (capped by smallest class for classification).

    Returns:
        JSON-safe dict with leaderboard rows, winner card, and holdout metrics.
    """
    if not target_column or target_column not in df.columns:
        return {"available": False, "message": f"Target '{target_column}' not in dataframe."}

    prep = prepare_ml_matrix(df, target=target_column)
    if not prep["usable"]:
        return {
            "available": False,
            "message": "; ".join(prep["warnings"]) or "Insufficient data for ML.",
            "dropped_columns": prep["dropped_columns"],
        }

    X = prep["X"]
    y = prep["y"]
    numeric_features = prep["numeric_features"]
    categorical_features = prep["categorical_features"]

    if task_type is None:
        task_type = _infer_task_type(y)

    # XGBoost requires integer-encoded labels for classification
    y_for_models = _coerce_classification_target(y) if task_type == "classification" else y

    if task_type == "classification" and y_for_models.nunique() < 2:
        return {"available": False, "message": "Target has <2 classes."}

    cv, actual_splits = _cv_for(task_type, y_for_models, n_splits=n_splits)
    scoring, primary = _scoring_for(task_type)

    if task_type == "classification":
        registry = _classification_registry(class_count=int(y_for_models.nunique()))
    else:
        registry = _regression_registry()

    leaderboard: list[dict] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for name, est in registry.items():
            preprocessor = build_sklearn_preprocessor(numeric_features, categorical_features)
            pipeline = Pipeline([("pre", preprocessor), ("model", est)])

            t0 = time.perf_counter()
            try:
                cv_results = cross_validate(
                    pipeline,
                    X,
                    y_for_models,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=False,
                    error_score=np.nan,
                )
                fit_time_s = float(np.mean(cv_results["fit_time"]))
                row: dict[str, Any] = {
                    "model": name,
                    "fit_time_s": _safe_round(fit_time_s, ndigits=3),
                    "cv_total_s": _safe_round(time.perf_counter() - t0, ndigits=3),
                    "n_splits": actual_splits,
                }
                for metric_key in scoring.keys():
                    arr = cv_results[f"test_{metric_key}"]
                    if metric_key.startswith("neg_"):
                        clean_key = metric_key[4:]
                        row[f"{clean_key}_mean"] = _safe_round(-np.nanmean(arr))
                        row[f"{clean_key}_std"] = _safe_round(np.nanstd(arr))
                    else:
                        row[f"{metric_key}_mean"] = _safe_round(np.nanmean(arr))
                        row[f"{metric_key}_std"] = _safe_round(np.nanstd(arr))
                row["primary_score"] = (
                    row.get(f"{primary}_mean")
                    if not primary.startswith("neg_")
                    else row.get(f"{primary[4:]}_mean")
                )
                leaderboard.append(row)
            except Exception as exc:
                leaderboard.append({"model": name, "error": str(exc)})

    # Sort by primary metric (higher is better for accuracy/f1/r2; for MAE/RMSE we use neg_)
    def _sort_key(row: dict) -> float:
        val = row.get("primary_score")
        return -float("inf") if val is None else float(val)

    leaderboard.sort(key=_sort_key, reverse=True)

    # Pick best non-dummy model
    candidates = [
        r for r in leaderboard
        if r.get("primary_score") is not None and "Dummy" not in r["model"]
    ]
    if not candidates:
        return {
            "available": True,
            "task_type": task_type,
            "target_column": target_column,
            "n_rows": int(len(y)),
            "n_features": len(numeric_features) + len(categorical_features),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "leaderboard": leaderboard,
            "winner": None,
            "message": "No non-dummy model produced a usable score.",
        }

    winner = candidates[0]

    # Refit winner on full data and run holdout
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        winner_estimator = registry[winner["model"]]
        winner_preprocessor = build_sklearn_preprocessor(numeric_features, categorical_features)
        winner_pipeline = Pipeline([("pre", winner_preprocessor), ("model", winner_estimator)])

        try:
            if task_type == "classification":
                holdout = _classification_holdout(winner_pipeline, X, y_for_models)
            else:
                holdout = _regression_holdout(winner_pipeline, X, y_for_models)
        except Exception as exc:
            holdout = {"available": False, "error": str(exc)}

    return {
        "available": True,
        "task_type": task_type,
        "target_column": target_column,
        "primary_metric": primary,
        "n_rows": int(len(y)),
        "n_features": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": prep["dropped_columns"],
        "warnings": prep["warnings"],
        "leaderboard": leaderboard,
        "winner": {
            "model": winner["model"],
            "primary_score": winner["primary_score"],
            "fit_time_s": winner["fit_time_s"],
            "holdout": holdout,
        },
    }
