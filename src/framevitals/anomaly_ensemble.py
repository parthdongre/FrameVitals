"""Multi-detector anomaly scoring for tabular data.

The ensemble combines deterministic/classical detectors that are useful across
many tabular datasets and optionally adds ECOD/COPOD when PyOD is installed.
Scores remain normalized to [0, 1] and the historical mean-ensemble threshold
is preserved, while the result now also reports detector agreement, failures,
input preparation, and feature-level context for top anomalous rows.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def _to_unit(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize an arbitrary score vector to [0, 1]."""
    s = np.asarray(scores, dtype=float)
    s = np.where(np.isfinite(s), s, np.nan)
    if np.all(np.isnan(s)):
        return np.zeros_like(s)
    lo = np.nanmin(s)
    hi = np.nanmax(s)
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    out = (s - lo) / (hi - lo)
    return np.where(np.isnan(out), 0.0, out)


def _prepare_numeric_matrix(
    df: pd.DataFrame,
    max_columns: int = 30,
) -> tuple[pd.DataFrame | None, list[str], dict[str, Any]]:
    """Select, sanitize, impute, and scale numeric columns."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    original_numeric_columns = list(numeric.columns)

    infinity_counts: dict[str, int] = {}
    for column in list(numeric.columns):
        values = pd.to_numeric(numeric[column], errors="coerce")
        finite_array = values.to_numpy(dtype="float64", na_value=np.nan)
        inf_count = int(np.isinf(finite_array).sum())
        if inf_count:
            infinity_counts[column] = inf_count
        numeric[column] = values.replace([np.inf, -np.inf], np.nan)

    dropped_all_missing: list[str] = []
    dropped_constant: list[str] = []
    for column in list(numeric.columns):
        if numeric[column].notna().sum() == 0:
            numeric = numeric.drop(columns=[column])
            dropped_all_missing.append(column)
        elif numeric[column].nunique(dropna=True) <= 1:
            numeric = numeric.drop(columns=[column])
            dropped_constant.append(column)

    if numeric.shape[1] < 1:
        metadata = {
            "numeric_columns_found": len(original_numeric_columns),
            "used_columns": [],
            "dropped_constant_columns": dropped_constant,
            "dropped_all_missing_columns": dropped_all_missing,
            "infinite_values_replaced": infinity_counts,
            "missing_values_imputed": {},
            "truncated_columns": False,
        }
        return None, [], metadata

    truncated = numeric.shape[1] > max_columns
    if truncated:
        keep = (
            numeric.notna()
            .sum()
            .sort_values(ascending=False)
            .head(max_columns)
            .index.tolist()
        )
        numeric = numeric[keep]

    missing_counts = {
        column: int(count)
        for column, count in numeric.isna().sum().items()
        if int(count) > 0
    }

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(numeric)
    X = scaler.fit_transform(X)

    used_columns = list(numeric.columns)
    metadata = {
        "numeric_columns_found": len(original_numeric_columns),
        "used_columns": used_columns,
        "dropped_constant_columns": dropped_constant,
        "dropped_all_missing_columns": dropped_all_missing,
        "infinite_values_replaced": {
            key: value for key, value in infinity_counts.items() if key in used_columns
        },
        "missing_values_imputed": missing_counts,
        "truncated_columns": truncated,
    }
    return pd.DataFrame(X, columns=used_columns, index=df.index), used_columns, metadata


def _detect_iforest(X: np.ndarray, contamination: float) -> np.ndarray:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return _to_unit(-model.score_samples(X))


def _detect_lof(X: np.ndarray) -> np.ndarray:
    n_neighbors = min(20, max(5, X.shape[0] // 20))
    n_neighbors = min(n_neighbors, max(X.shape[0] - 1, 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, n_jobs=-1)
    model.fit_predict(X)
    return _to_unit(-model.negative_outlier_factor_)


def _detect_elliptic(X: np.ndarray, contamination: float) -> np.ndarray:
    model = EllipticEnvelope(
        contamination=contamination,
        support_fraction=None,
        random_state=42,
    )
    model.fit(X)
    return _to_unit(-model.score_samples(X))


def _detect_mad_robust_z(X: np.ndarray) -> np.ndarray:
    """Per-row mean of robust z-scores across columns."""
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad_safe = np.where(mad > 0, mad, 1.0)
    z = np.abs((X - median) / (1.4826 * mad_safe))
    return _to_unit(z.mean(axis=1))


def _detect_mahalanobis(X: np.ndarray) -> np.ndarray:
    """Mahalanobis distance using robust covariance with a stable fallback."""
    try:
        cov = MinCovDet(random_state=42).fit(X)
        raw = cov.mahalanobis(X)
        return _to_unit(raw)
    except Exception:
        mean = X.mean(axis=0)
        cov_matrix = np.atleast_2d(np.cov(X.T))
        cov_matrix = cov_matrix + np.eye(X.shape[1]) * 1e-6
        inv = np.linalg.pinv(cov_matrix)
        diff = X - mean
        raw = np.einsum("ij,jk,ik->i", diff, inv, diff)
        return _to_unit(np.sqrt(np.maximum(raw, 0)))


def _detect_pyod(X: np.ndarray, model_name: str) -> np.ndarray:
    if model_name == "ecod":
        from pyod.models.ecod import ECOD as Model
    elif model_name == "copod":
        from pyod.models.copod import COPOD as Model
    else:
        raise ValueError(f"Unsupported PyOD detector: {model_name}")

    model = Model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)
    return _to_unit(model.decision_scores_)


def _detector_summary(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    return {
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "p95": round(float(np.quantile(values, 0.95)), 4),
        "p99": round(float(np.quantile(values, 0.99)), 4),
        "max": round(float(np.max(values)), 4),
    }


def _top_feature_deviations(
    X_df: pd.DataFrame,
    row_index: Any,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    row = X_df.loc[row_index]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    ordered = row.abs().sort_values(ascending=False).head(limit)
    return [
        {
            "feature": str(feature),
            "standardized_deviation": round(float(abs(row[feature])), 4),
            "direction": "high" if float(row[feature]) >= 0 else "low",
        }
        for feature in ordered.index
    ]


def detect_anomalies_ensemble(
    df: pd.DataFrame,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
) -> dict[str, Any]:
    """Run available anomaly detectors and return an explainable JSON-safe summary."""
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    contamination = float(np.clip(contamination, 0.001, 0.5))
    X_df, used_cols, preparation = _prepare_numeric_matrix(
        df,
        max_columns=max_columns,
    )
    if X_df is None or X_df.empty:
        return {
            "available": False,
            "message": "No usable numeric columns (need at least 1 non-constant numeric column).",
            "preparation": preparation,
        }

    X = X_df.values
    n_rows = X.shape[0]
    if n_rows < 20:
        return {
            "available": False,
            "message": "Need at least 20 rows for anomaly ensemble.",
            "used_columns": used_cols,
            "preparation": preparation,
        }

    detector_scores: dict[str, np.ndarray] = {}
    detectors_failed: dict[str, str] = {}
    detectors_skipped: dict[str, str] = {}

    def run_detector(name: str, fn: Callable[[], np.ndarray]) -> None:
        try:
            scores = np.asarray(fn(), dtype=float)
            if scores.shape != (n_rows,):
                raise ValueError(
                    f"Detector returned shape {scores.shape}; expected {(n_rows,)}."
                )
            if not np.isfinite(scores).all():
                raise ValueError("Detector produced non-finite scores.")
            detector_scores[name] = scores
        except ImportError:
            detectors_skipped[name] = "Optional dependency is not installed."
        except Exception as exc:  # noqa: BLE001 - individual detectors fail soft
            detectors_failed[name] = f"{type(exc).__name__}: {exc}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_detector(
            "isolation_forest",
            lambda: _detect_iforest(X, contamination),
        )
        run_detector("local_outlier_factor", lambda: _detect_lof(X))

        if n_rows >= max(20, X.shape[1] * 2 + 1):
            run_detector(
                "elliptic_envelope",
                lambda: _detect_elliptic(X, contamination),
            )
        else:
            detectors_skipped["elliptic_envelope"] = (
                "Too few rows relative to numeric dimensionality for stable covariance estimation."
            )

        run_detector("mad_robust_z", lambda: _detect_mad_robust_z(X))
        run_detector("mahalanobis", lambda: _detect_mahalanobis(X))
        run_detector("ecod", lambda: _detect_pyod(X, "ecod"))
        run_detector("copod", lambda: _detect_pyod(X, "copod"))

    if not detector_scores:
        return {
            "available": False,
            "message": "All anomaly detectors failed or were unavailable.",
            "used_columns": used_cols,
            "preparation": preparation,
            "detectors_failed": detectors_failed,
            "detectors_skipped": detectors_skipped,
        }

    detector_names = list(detector_scores)
    score_matrix = np.column_stack([detector_scores[name] for name in detector_names])
    ensemble = score_matrix.mean(axis=1)

    score_df = pd.DataFrame(
        {name: detector_scores[name] for name in detector_names},
        index=df.index,
    )
    score_df["ensemble"] = ensemble

    vote_thresholds: dict[str, float] = {}
    vote_matrix: list[np.ndarray] = []
    for name in detector_names:
        detector_threshold = float(
            np.quantile(detector_scores[name], 1.0 - contamination)
        )
        vote_thresholds[name] = detector_threshold
        vote_matrix.append(detector_scores[name] >= detector_threshold)

    agreement_count = np.column_stack(vote_matrix).sum(axis=1)
    agreement_fraction = agreement_count / len(detector_names)
    score_df["agreement_count"] = agreement_count
    score_df["agreement_fraction"] = agreement_fraction

    majority_required = max(1, math.ceil(len(detector_names) / 2))
    consensus_mask = agreement_count >= majority_required
    flagged_mask = ensemble >= threshold

    es = ensemble[np.isfinite(ensemble)]
    summary = {
        "min": float(np.min(es)),
        "mean": float(np.mean(es)),
        "median": float(np.median(es)),
        "max": float(np.max(es)),
        "p95": float(np.quantile(es, 0.95)),
        "p99": float(np.quantile(es, 0.99)),
    }

    flagged_count = int(flagged_mask.sum())
    consensus_flagged_count = int(consensus_mask.sum())

    top = score_df.sort_values("ensemble", ascending=False).head(top_k)
    top_rows: list[dict[str, Any]] = []
    for idx, row in top.iterrows():
        entry: dict[str, Any] = {
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
        }
        for name in detector_names:
            entry[name] = round(float(row[name]), 4)
        entry["ensemble"] = round(float(row["ensemble"]), 4)
        entry["flagged"] = bool(row["ensemble"] >= threshold)
        entry["agreement_count"] = int(row["agreement_count"])
        entry["agreement_fraction"] = round(float(row["agreement_fraction"]), 4)
        entry["top_feature_deviations"] = _top_feature_deviations(X_df, idx)
        top_rows.append(entry)

    detector_summaries = {
        name: _detector_summary(scores)
        for name, scores in detector_scores.items()
    }

    return {
        "available": True,
        "n_rows_scored": int(n_rows),
        "used_columns": used_cols,
        "preparation": preparation,
        "detectors_run": detector_names,
        "detectors_failed": detectors_failed,
        "detectors_skipped": detectors_skipped,
        "detector_summaries": detector_summaries,
        "detector_vote_thresholds": {
            name: round(value, 4) for name, value in vote_thresholds.items()
        },
        "threshold": float(threshold),
        "contamination": float(contamination),
        "expected_anomaly_count": int(math.ceil(n_rows * contamination)),
        "flagged_count": flagged_count,
        "flagged_fraction": round(float(flagged_count / n_rows), 4),
        "consensus": {
            "majority_detectors_required": majority_required,
            "flagged_count": consensus_flagged_count,
            "flagged_fraction": round(float(consensus_flagged_count / n_rows), 4),
        },
        "ensemble_summary": {k: round(v, 4) for k, v in summary.items()},
        "top_rows": top_rows,
        "interpretation": (
            "Each available detector emits a normalized [0,1] anomaly score; the historical "
            "ensemble remains their per-row mean. The configured threshold controls the main "
            "flag, while detector agreement independently reports how many detectors place a row "
            "in their contamination-adjusted anomaly tail. Top feature deviations are standardized "
            "context, not causal explanations."
        ),
    }
