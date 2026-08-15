"""
Anomaly Ensemble (WS-2)
=======================
Multi-detector anomaly scoring for tabular data.

Detectors:
    1. IsolationForest          (sklearn.ensemble)
    2. LocalOutlierFactor       (sklearn.neighbors)
    3. EllipticEnvelope         (sklearn.covariance) — robust Gaussian
    4. Robust z-score (MAD)     (numpy)
    5. Mahalanobis distance     (sklearn shrinkage covariance)
    6. ECOD                     (pyod, optional)
    7. COPOD                    (pyod, optional)

Each detector emits a normalized score in [0, 1] (higher = more anomalous).
The ensemble score is the column-wise mean across available detectors.

Public entry point:
    detect_anomalies_ensemble(df, contamination=0.05) -> dict
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    out = np.where(np.isnan(out), 0.0, out)
    return out


def _prepare_numeric_matrix(df: pd.DataFrame, max_columns: int = 30):
    """
    Select numeric columns, drop constants, impute medians, scale.
    Returns (X_scaled_df, used_columns) or (None, []) if not enough columns.
    """
    numeric = df.select_dtypes(include=[np.number]).copy()

    # Drop constants
    for col in list(numeric.columns):
        if numeric[col].nunique(dropna=True) <= 1:
            numeric = numeric.drop(columns=[col])

    if numeric.shape[1] < 1:
        return None, []

    # Cap dimensionality (keep most-populated columns)
    if numeric.shape[1] > max_columns:
        keep = (
            numeric.notna().sum().sort_values(ascending=False).head(max_columns).index.tolist()
        )
        numeric = numeric[keep]

    # Median impute then scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = imputer.fit_transform(numeric)
    X = scaler.fit_transform(X)

    return pd.DataFrame(X, columns=numeric.columns, index=df.index), list(numeric.columns)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _detect_iforest(X: np.ndarray, contamination: float) -> np.ndarray:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    # score_samples: higher = more normal. Negate so higher = more anomalous.
    raw = -model.score_samples(X)
    return _to_unit(raw)


def _detect_lof(X: np.ndarray) -> np.ndarray:
    model = LocalOutlierFactor(n_neighbors=min(20, max(5, X.shape[0] // 20)), n_jobs=-1)
    model.fit_predict(X)
    raw = -model.negative_outlier_factor_  # higher = more anomalous
    return _to_unit(raw)


def _detect_elliptic(X: np.ndarray, contamination: float) -> np.ndarray:
    try:
        model = EllipticEnvelope(contamination=contamination, support_fraction=None, random_state=42)
        model.fit(X)
        raw = -model.score_samples(X)
        return _to_unit(raw)
    except Exception:
        return np.zeros(X.shape[0])


def _detect_mad_robust_z(X: np.ndarray) -> np.ndarray:
    """Per-row mean of robust z-scores across columns."""
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    mad_safe = np.where(mad > 0, mad, 1.0)
    z = np.abs((X - median) / (1.4826 * mad_safe))
    raw = z.mean(axis=1)
    return _to_unit(raw)


def _detect_mahalanobis(X: np.ndarray) -> np.ndarray:
    """Mahalanobis distance using a shrinkage / robust covariance estimate."""
    try:
        cov = MinCovDet(random_state=42).fit(X)
        raw = cov.mahalanobis(X)
        return _to_unit(raw)
    except Exception:
        # Fallback to plain covariance
        try:
            mean = X.mean(axis=0)
            cov_matrix = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
            inv = np.linalg.pinv(cov_matrix)
            diff = X - mean
            raw = np.einsum("ij,jk,ik->i", diff, inv, diff)
            return _to_unit(np.sqrt(np.maximum(raw, 0)))
        except Exception:
            return np.zeros(X.shape[0])


def _detect_pyod(X: np.ndarray, model_name: str) -> np.ndarray | None:
    """Try ECOD or COPOD from pyod; return None if pyod is not installed."""
    try:
        if model_name == "ecod":
            from pyod.models.ecod import ECOD as Model
        elif model_name == "copod":
            from pyod.models.copod import COPOD as Model
        else:
            return None
        model = Model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
        return _to_unit(model.decision_scores_)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_anomalies_ensemble(
    df: pd.DataFrame,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
) -> dict:
    """
    Run all available detectors and return a JSON-safe summary.

    Args:
        df: Input dataframe.
        contamination: Expected anomaly fraction passed to detectors that need it.
        threshold: Ensemble score above which a row is flagged.
        max_columns: Cap on numeric dimensionality.
        top_k: Number of top rows to include in the output payload.

    Returns shape:
        {
            "available": bool,
            "n_rows_scored": int,
            "used_columns": [...],
            "detectors_run": [...],
            "threshold": float,
            "flagged_count": int,
            "ensemble_summary": {"min","mean","median","max","p95","p99"},
            "top_rows": [{"row_index", "ensemble", "<detector>": score, ...}],
            "score_table": pandas-records (top_k rows only),
        }
    """
    X_df, used_cols = _prepare_numeric_matrix(df, max_columns=max_columns)
    if X_df is None or X_df.empty:
        return {
            "available": False,
            "message": "No usable numeric columns (need at least 1 non-constant numeric column).",
        }

    X = X_df.values
    n_rows = X.shape[0]
    if n_rows < 20:
        return {
            "available": False,
            "message": "Need at least 20 rows for anomaly ensemble.",
            "used_columns": used_cols,
        }

    # Bound contamination
    contamination = float(np.clip(contamination, 0.001, 0.5))

    detector_scores: dict[str, np.ndarray] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            detector_scores["isolation_forest"] = _detect_iforest(X, contamination)
        except Exception:
            pass

        try:
            detector_scores["local_outlier_factor"] = _detect_lof(X)
        except Exception:
            pass

        try:
            elliptic = _detect_elliptic(X, contamination)
            if elliptic.any():
                detector_scores["elliptic_envelope"] = elliptic
        except Exception:
            pass

        try:
            detector_scores["mad_robust_z"] = _detect_mad_robust_z(X)
        except Exception:
            pass

        try:
            mahal = _detect_mahalanobis(X)
            if mahal.any():
                detector_scores["mahalanobis"] = mahal
        except Exception:
            pass

        ecod = _detect_pyod(X, "ecod")
        if ecod is not None:
            detector_scores["ecod"] = ecod

        copod = _detect_pyod(X, "copod")
        if copod is not None:
            detector_scores["copod"] = copod

    if not detector_scores:
        return {
            "available": False,
            "message": "All detectors failed.",
            "used_columns": used_cols,
        }

    # Stack detector scores into a (n_rows, n_detectors) matrix and average
    score_matrix = np.column_stack(list(detector_scores.values()))
    ensemble = score_matrix.mean(axis=1)

    # Build per-row table (in original df index)
    score_df = pd.DataFrame(
        {name: scores for name, scores in detector_scores.items()},
        index=df.index,
    )
    score_df["ensemble"] = ensemble

    # Summary stats
    es = ensemble[np.isfinite(ensemble)]
    summary = {
        "min": float(np.min(es)),
        "mean": float(np.mean(es)),
        "median": float(np.median(es)),
        "max": float(np.max(es)),
        "p95": float(np.quantile(es, 0.95)),
        "p99": float(np.quantile(es, 0.99)),
    }

    flagged_mask = score_df["ensemble"] >= threshold
    flagged_count = int(flagged_mask.sum())

    # Top-k rows
    top = score_df.sort_values("ensemble", ascending=False).head(top_k)
    top_rows = []
    for idx, row in top.iterrows():
        entry = {"row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx)}
        for name in detector_scores.keys():
            entry[name] = round(float(row[name]), 4)
        entry["ensemble"] = round(float(row["ensemble"]), 4)
        top_rows.append(entry)

    return {
        "available": True,
        "n_rows_scored": int(n_rows),
        "used_columns": used_cols,
        "detectors_run": list(detector_scores.keys()),
        "threshold": float(threshold),
        "contamination": float(contamination),
        "flagged_count": flagged_count,
        "ensemble_summary": {k: round(v, 4) for k, v in summary.items()},
        "top_rows": top_rows,
        "interpretation": (
            "Each detector emits a [0,1] anomaly score; the ensemble is the per-row mean "
            "across all available detectors. Rows with ensemble >= "
            f"{threshold} are flagged."
        ),
    }
