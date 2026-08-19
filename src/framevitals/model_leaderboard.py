"""Cross-validated baseline model leaderboard for FrameVitals.

The leaderboard is a diagnostic for dataset learnability, not an AutoML system.
It compares several lightweight/classical models (plus optional XGBoost and
LightGBM), keeps preprocessing inside each CV fold, records failures, and
compares the best real model against a dummy baseline.
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


def _infer_task_type(y: pd.Series) -> str:
    """Best-effort: classification if dtype is non-numeric or low-cardinality."""
    if pd.api.types.is_numeric_dtype(y):
        n = len(y)
        unique = int(y.nunique(dropna=True))
        if unique <= 20 and unique <= max(2, int(n * 0.05)):
            return "classification"
        return "regression"
    return "classification"


def _classification_registry(class_count: int) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
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
            n_estimators=150,
            max_depth=4,
            random_state=42,
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
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            random_state=42,
        ),
    }
    xgb = _maybe_xgb_regressor()
    if xgb is not None:
        registry["XGBRegressor"] = xgb
    lgbm = _maybe_lgbm_regressor()
    if lgbm is not None:
        registry["LGBMRegressor"] = lgbm
    return registry


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
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    n_rows = len(y)
    if n_rows < 2:
        raise ValueError("At least 2 target rows are required for cross-validation.")

    if task_type == "classification":
        counts = y.value_counts()
        min_class = int(counts.min()) if len(counts) else 0
        if min_class < 2:
            raise ValueError(
                "Each classification class needs at least 2 rows for stratified cross-validation."
            )
        actual_splits = min(n_splits, min_class)
        return (
            StratifiedKFold(
                n_splits=actual_splits,
                shuffle=True,
                random_state=42,
            ),
            actual_splits,
        )

    # R² is undefined for a test fold containing fewer than two observations.
    # Because ML preprocessing already requires >=20 rows, capping KFold at
    # floor(n/2) guarantees every test fold has at least two rows while still
    # honouring smaller user-requested fold counts.
    max_r2_splits = max(2, n_rows // 2)
    actual_splits = min(n_splits, max_r2_splits)
    if actual_splits < 2:
        raise ValueError("Regression cross-validation needs at least 2 folds.")
    return (
        KFold(n_splits=actual_splits, shuffle=True, random_state=42),
        actual_splits,
    )


def _scoring_for(task_type: str) -> tuple[dict[str, str], str]:
    if task_type == "classification":
        return {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
        }, "f1_weighted"
    return {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
    }, "r2"


def _json_safe_label(value: Any) -> Any:
    """Preserve ordinary class-label types while keeping metadata JSON-safe."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _encode_classification_target(
    y: pd.Series,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Encode every classification target to stable consecutive integer labels."""
    # sort=False also supports heterogeneous object labels that cannot be
    # ordered against one another (for example a mix of strings and integers).
    codes, uniques = pd.factorize(y, sort=False)
    encoded = pd.Series(codes, index=y.index, name=y.name, dtype="int64")
    mapping = [
        {
            "encoded": int(index),
            "label": _json_safe_label(value),
        }
        for index, value in enumerate(uniques.tolist())
    ]
    return encoded, mapping


def _set_estimator_jobs(estimator: Any, jobs: int) -> Any:
    """Avoid nested process/thread explosions inside parallel CV folds."""
    try:
        params = estimator.get_params(deep=False)
        if "n_jobs" in params:
            estimator.set_params(n_jobs=jobs)
    except Exception:
        pass
    return estimator


def _adapt_knn_neighbors(registry: dict[str, Any], n_rows: int, n_splits: int) -> int:
    largest_test_fold = int(np.ceil(n_rows / n_splits))
    smallest_train_fold = max(1, n_rows - largest_test_fold)
    neighbors = max(1, min(7, smallest_train_fold))
    for name in ("KNeighborsClassifier", "KNeighborsRegressor"):
        estimator = registry.get(name)
        if estimator is not None:
            try:
                estimator.set_params(n_neighbors=neighbors)
            except Exception:
                pass
    return neighbors


def _score_stability(std: float | None) -> str:
    if std is None:
        return "unknown"
    if std <= 0.02:
        return "stable"
    if std <= 0.05:
        return "moderate"
    return "variable"


def _classification_holdout(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    n = len(y)
    test_size = max(50, int(n * 0.25)) if n >= 200 else max(20, int(n * 0.25))
    test_size = min(test_size, n - 20)
    if test_size <= 0:
        return {"available": False, "reason": "not enough rows for holdout"}

    class_count = int(y.nunique())
    stratify = y if class_count > 1 and y.value_counts().min() >= 2 else None
    if stratify is not None:
        test_size = max(test_size, class_count)
        if n - test_size < class_count:
            stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    out: dict[str, Any] = {
        "available": True,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": _safe_round(accuracy_score(y_test, y_pred)),
        "f1_weighted": _safe_round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "precision_weighted": _safe_round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": _safe_round(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }

    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_test)
            classes = pipeline.classes_ if hasattr(pipeline, "classes_") else None
            if proba.shape[1] == 2 and classes is not None:
                pos_idx = 1
                y_bin = (y_test == classes[pos_idx]).astype(int).values
                out["brier_score"] = _safe_round(
                    brier_score_loss(y_bin, proba[:, pos_idx])
                )
    except Exception as exc:
        out["brier_error"] = str(exc)

    return out


def _regression_holdout(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:
    n = len(y)
    test_size = max(50, int(n * 0.25)) if n >= 200 else max(20, int(n * 0.25))
    test_size = min(test_size, n - 20)
    if test_size <= 0:
        return {"available": False, "reason": "not enough rows for holdout"}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
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


def run_model_leaderboard(
    df: pd.DataFrame,
    target_column: str,
    task_type: str | None = None,
    n_splits: int = 5,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Run a CV-validated diagnostic model leaderboard."""
    if not target_column or target_column not in df.columns:
        return {
            "available": False,
            "message": f"Target '{target_column}' not in dataframe.",
        }
    if task_type is not None and task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification', 'regression', or None.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0.")

    prep = prepare_ml_matrix(df, target=target_column)
    if not prep["usable"]:
        return {
            "available": False,
            "message": "; ".join(prep["warnings"]) or "Insufficient data for ML.",
            "dropped_columns": prep["dropped_columns"],
            "warnings": prep["warnings"],
        }

    X = prep["X"]
    y = prep["y"]
    numeric_features = prep["numeric_features"]
    categorical_features = prep["categorical_features"]

    if task_type is None:
        task_type = _infer_task_type(y)

    target_encoding: list[dict[str, Any]] | None = None
    if task_type == "classification":
        y_for_models, target_encoding = _encode_classification_target(y)
        if y_for_models.nunique() < 2:
            return {"available": False, "message": "Target has <2 classes."}
    else:
        if not pd.api.types.is_numeric_dtype(y):
            return {
                "available": False,
                "message": "Regression requires a numeric target.",
            }
        y_for_models = pd.to_numeric(y, errors="coerce")
        if not np.isfinite(y_for_models.to_numpy(dtype=float)).all():
            return {
                "available": False,
                "message": "Regression target contains non-finite values after preprocessing.",
            }

    try:
        cv, actual_splits = _cv_for(
            task_type,
            y_for_models,
            n_splits=n_splits,
        )
    except ValueError as exc:
        return {
            "available": False,
            "task_type": task_type,
            "target_column": target_column,
            "message": str(exc),
        }

    scoring, primary = _scoring_for(task_type)
    if task_type == "classification":
        registry = _classification_registry(
            class_count=int(y_for_models.nunique())
        )
    else:
        registry = _regression_registry()

    knn_neighbors = _adapt_knn_neighbors(
        registry,
        n_rows=len(y_for_models),
        n_splits=actual_splits,
    )
    for estimator in registry.values():
        _set_estimator_jobs(estimator, 1)

    leaderboard: list[dict[str, Any]] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for name, estimator in registry.items():
            preprocessor = build_sklearn_preprocessor(
                numeric_features,
                categorical_features,
            )
            pipeline = Pipeline([
                ("pre", preprocessor),
                ("model", estimator),
            ])

            t0 = time.perf_counter()
            try:
                cv_results = cross_validate(
                    pipeline,
                    X,
                    y_for_models,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    return_train_score=False,
                    error_score=np.nan,
                )
                row: dict[str, Any] = {
                    "model": name,
                    "fit_time_s": _safe_round(
                        np.nanmean(cv_results["fit_time"]),
                        ndigits=3,
                    ),
                    "cv_total_s": _safe_round(
                        time.perf_counter() - t0,
                        ndigits=3,
                    ),
                    "n_splits": actual_splits,
                }
                for metric_key in scoring:
                    arr = np.asarray(cv_results[f"test_{metric_key}"], dtype=float)
                    valid_folds = int(np.isfinite(arr).sum())
                    if metric_key.startswith("neg_"):
                        clean_key = metric_key[4:]
                        row[f"{clean_key}_mean"] = _safe_round(-np.nanmean(arr))
                        row[f"{clean_key}_std"] = _safe_round(np.nanstd(arr))
                        row[f"{clean_key}_valid_folds"] = valid_folds
                    else:
                        row[f"{metric_key}_mean"] = _safe_round(np.nanmean(arr))
                        row[f"{metric_key}_std"] = _safe_round(np.nanstd(arr))
                        row[f"{metric_key}_valid_folds"] = valid_folds

                row["primary_score"] = row.get(f"{primary}_mean")
                row["primary_std"] = row.get(f"{primary}_std")
                row["score_stability"] = _score_stability(row["primary_std"])
                if row["primary_score"] is None:
                    row["error"] = "No valid cross-validation score was produced."
                leaderboard.append(row)
            except Exception as exc:
                leaderboard.append({
                    "model": name,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    def _sort_key(row: dict[str, Any]) -> float:
        value = row.get("primary_score")
        return -float("inf") if value is None else float(value)

    leaderboard.sort(key=_sort_key, reverse=True)

    successful_rows = [
        row for row in leaderboard if row.get("primary_score") is not None
    ]
    failed_rows = [row for row in leaderboard if row.get("primary_score") is None]
    dummy_row = next(
        (row for row in successful_rows if "Dummy" in row["model"]),
        None,
    )
    candidates = [
        row for row in successful_rows if "Dummy" not in row["model"]
    ]

    base_payload: dict[str, Any] = {
        "available": True,
        "task_type": task_type,
        "target_column": target_column,
        "primary_metric": primary,
        "n_rows": int(len(y_for_models)),
        "n_features": len(numeric_features) + len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": prep["dropped_columns"],
        "warnings": list(prep["warnings"]),
        "target_encoding": target_encoding,
        "cv": {
            "requested_splits": int(n_splits),
            "actual_splits": int(actual_splits),
            "n_jobs": int(n_jobs),
            "knn_neighbors": int(knn_neighbors),
        },
        "models_succeeded": len(successful_rows),
        "models_failed": len(failed_rows),
        "model_failures": [
            {"model": row["model"], "error": row.get("error")}
            for row in failed_rows
        ],
        "leaderboard": leaderboard,
        "baseline": (
            {
                "model": dummy_row["model"],
                "primary_score": dummy_row["primary_score"],
            }
            if dummy_row is not None
            else None
        ),
    }

    if not candidates:
        return {
            **base_payload,
            "winner": None,
            "message": "No non-dummy model produced a usable score.",
        }

    winner = candidates[0]
    baseline_score = (
        float(dummy_row["primary_score"])
        if dummy_row is not None and dummy_row.get("primary_score") is not None
        else None
    )
    winner_score = float(winner["primary_score"])
    lift = winner_score - baseline_score if baseline_score is not None else None
    beats_baseline = bool(lift is not None and lift > 0)

    if baseline_score is not None and not beats_baseline:
        base_payload["warnings"].append(
            "Best non-dummy model did not outperform the dummy baseline on the primary CV metric."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        winner_estimator = registry[winner["model"]]
        winner_preprocessor = build_sklearn_preprocessor(
            numeric_features,
            categorical_features,
        )
        winner_pipeline = Pipeline([
            ("pre", winner_preprocessor),
            ("model", winner_estimator),
        ])

        try:
            if task_type == "classification":
                holdout = _classification_holdout(
                    winner_pipeline,
                    X,
                    y_for_models,
                )
            else:
                holdout = _regression_holdout(
                    winner_pipeline,
                    X,
                    y_for_models,
                )
        except Exception as exc:
            holdout = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        **base_payload,
        "winner": {
            "model": winner["model"],
            "primary_score": winner["primary_score"],
            "primary_std": winner.get("primary_std"),
            "score_stability": winner.get("score_stability"),
            "fit_time_s": winner.get("fit_time_s"),
            "baseline_score": _safe_round(baseline_score),
            "lift_over_baseline": _safe_round(lift),
            "beats_baseline": beats_baseline if baseline_score is not None else None,
            "holdout": holdout,
        },
    }
