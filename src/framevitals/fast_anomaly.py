"""Fast bounded anomaly screening for standard and deep analysis modes.

The scanner combines robust per-feature deviation, sparse random-projection tail
scores, and projection-density rarity. It is fully vectorized and avoids tree,
nearest-neighbour, and covariance fitting. Research mode can still run the
heavier classical/neural ensemble for confirmation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _to_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, np.nan)
    if finite.size == 0 or np.all(np.isnan(finite)):
        return np.zeros_like(values, dtype=np.float64)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    out = (finite - lo) / (hi - lo)
    return np.where(np.isfinite(out), out, 0.0)


def _summary(scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "mean": round(float(np.mean(scores)), 4),
        "median": round(float(np.median(scores)), 4),
        "p95": round(float(np.quantile(scores, 0.95)), 4),
        "p99": round(float(np.quantile(scores, 0.99)), 4),
        "max": round(float(np.max(scores)), 4),
    }


def _prepare_numeric(
    dataframe: pd.DataFrame,
    *,
    max_columns: int,
) -> tuple[np.ndarray | None, list[str], dict[str, Any]]:
    numeric = dataframe.select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    original_columns = list(numeric.columns)
    dropped_all_missing: list[str] = []
    dropped_constant: list[str] = []

    for column in list(numeric.columns):
        valid = numeric[column].dropna()
        if valid.empty:
            numeric = numeric.drop(columns=[column])
            dropped_all_missing.append(str(column))
        elif valid.nunique() <= 1:
            numeric = numeric.drop(columns=[column])
            dropped_constant.append(str(column))

    if numeric.empty:
        return None, [], {
            "numeric_columns_found": len(original_columns),
            "used_columns": [],
            "dropped_all_missing_columns": dropped_all_missing,
            "dropped_constant_columns": dropped_constant,
            "truncated_columns": False,
        }

    # Prefer columns with usable coverage and variation. This keeps ID-like or
    # mostly-empty columns from crowding out informative dimensions.
    variance = numeric.var(axis=0, skipna=True).fillna(0.0)
    coverage = numeric.notna().mean(axis=0)
    rank = (np.log1p(variance.abs()) + coverage).sort_values(ascending=False)
    selected = rank.head(max_columns).index.tolist()
    numeric = numeric[selected]

    medians = numeric.median(axis=0, skipna=True)
    numeric = numeric.fillna(medians).fillna(0.0)
    matrix = numeric.to_numpy(dtype=np.float64)

    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0)
    std = np.std(matrix, axis=0, ddof=1)
    scale = 1.4826 * mad
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    standardized = (matrix - center) / scale

    return standardized, [str(column) for column in selected], {
        "numeric_columns_found": len(original_columns),
        "used_columns": [str(column) for column in selected],
        "dropped_all_missing_columns": dropped_all_missing,
        "dropped_constant_columns": dropped_constant,
        "truncated_columns": len(selected) < len(original_columns) - len(dropped_all_missing) - len(dropped_constant),
        "scaling": "median_mad_with_std_fallback",
    }


def _projection_density_scores(projected: np.ndarray, bins: int = 32) -> np.ndarray:
    n_rows, n_projections = projected.shape
    if n_rows == 0 or n_projections == 0:
        return np.zeros(n_rows, dtype=np.float64)
    rarity = np.zeros((n_rows, n_projections), dtype=np.float64)

    for index in range(n_projections):
        values = projected[:, index]
        counts, edges = np.histogram(values, bins=min(bins, max(4, int(np.sqrt(n_rows)))))
        positions = np.searchsorted(edges, values, side="right") - 1
        positions = np.clip(positions, 0, len(counts) - 1)
        row_counts = counts[positions]
        rarity[:, index] = -np.log((row_counts + 1.0) / (n_rows + len(counts)))

    return _to_unit(np.mean(rarity, axis=1))


def fast_anomaly_scan(
    dataframe: pd.DataFrame,
    *,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 24,
    projections: int = 12,
    top_k: int = 25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return a fast explainable anomaly screen over a bounded dataframe."""
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")
    if projections < 2:
        raise ValueError("projections must be at least 2.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    contamination = float(np.clip(contamination, 0.001, 0.5))

    matrix, used_columns, preparation = _prepare_numeric(
        dataframe,
        max_columns=max_columns,
    )
    if matrix is None or len(matrix) < 20:
        return {
            "available": False,
            "message": "Need at least 20 rows and one non-constant numeric column.",
            "preparation": preparation,
        }

    n_rows, n_features = matrix.shape
    robust_abs = np.abs(matrix)
    top_feature_count = min(3, n_features)
    if top_feature_count == n_features:
        robust_raw = robust_abs.mean(axis=1)
    else:
        top = np.partition(robust_abs, n_features - top_feature_count, axis=1)[
            :, -top_feature_count:
        ]
        robust_raw = top.mean(axis=1)
    robust_score = _to_unit(robust_raw)

    rng = np.random.default_rng(random_state)
    projection_matrix = rng.choice(
        np.array([-1.0, 0.0, 1.0]),
        size=(n_features, projections),
        p=[0.25, 0.50, 0.25],
    )
    for index in range(projections):
        nonzero = np.count_nonzero(projection_matrix[:, index])
        if nonzero == 0:
            projection_matrix[index % n_features, index] = 1.0
            nonzero = 1
        projection_matrix[:, index] /= math.sqrt(nonzero)

    projected = matrix @ projection_matrix
    projection_center = np.median(projected, axis=0)
    projection_mad = np.median(np.abs(projected - projection_center), axis=0)
    projection_scale = np.where(1.4826 * projection_mad > 1e-12, 1.4826 * projection_mad, 1.0)
    projection_z = np.abs((projected - projection_center) / projection_scale)
    projection_top = np.partition(
        projection_z,
        max(0, projections - min(3, projections)),
        axis=1,
    )[:, -min(3, projections):]
    projection_tail_score = _to_unit(projection_top.mean(axis=1))
    density_score = _projection_density_scores(projected)

    detector_scores = {
        "robust_feature_deviation": robust_score,
        "random_projection_tail": projection_tail_score,
        "random_projection_density": density_score,
    }
    ensemble = (
        0.50 * robust_score
        + 0.35 * projection_tail_score
        + 0.15 * density_score
    )
    ensemble = _to_unit(ensemble)

    detector_names = list(detector_scores)
    vote_thresholds = {
        name: float(np.quantile(scores, 1.0 - contamination))
        for name, scores in detector_scores.items()
    }
    votes = np.column_stack([
        detector_scores[name] >= vote_thresholds[name]
        for name in detector_names
    ])
    agreement_count = votes.sum(axis=1)
    majority_required = max(1, math.ceil(len(detector_names) / 2))
    consensus_mask = agreement_count >= majority_required
    flagged_mask = ensemble >= threshold

    top_indices = np.argsort(ensemble)[::-1][: min(top_k, n_rows)]
    top_rows: list[dict[str, Any]] = []
    for position in top_indices:
        feature_order = np.argsort(robust_abs[position])[::-1][: min(3, n_features)]
        top_rows.append({
            "row_index": (
                int(dataframe.index[position])
                if isinstance(dataframe.index[position], (int, np.integer))
                else str(dataframe.index[position])
            ),
            "robust_feature_deviation": round(float(robust_score[position]), 4),
            "random_projection_tail": round(float(projection_tail_score[position]), 4),
            "random_projection_density": round(float(density_score[position]), 4),
            "ensemble": round(float(ensemble[position]), 4),
            "flagged": bool(flagged_mask[position]),
            "agreement_count": int(agreement_count[position]),
            "agreement_fraction": round(float(agreement_count[position] / len(detector_names)), 4),
            "top_feature_deviations": [
                {
                    "feature": used_columns[int(feature_index)],
                    "standardized_deviation": round(float(robust_abs[position, feature_index]), 4),
                    "direction": "high" if matrix[position, feature_index] >= 0 else "low",
                }
                for feature_index in feature_order
            ],
        })

    flagged_count = int(flagged_mask.sum())
    consensus_count = int(consensus_mask.sum())
    return {
        "available": True,
        "method": "fast_robust_random_projection",
        "n_rows_scored": int(n_rows),
        "used_columns": used_columns,
        "preparation": preparation,
        "detectors_run": detector_names,
        "detectors_failed": {},
        "detectors_skipped": {},
        "detector_summaries": {
            name: _summary(scores) for name, scores in detector_scores.items()
        },
        "detector_vote_thresholds": {
            name: round(value, 4) for name, value in vote_thresholds.items()
        },
        "threshold": float(threshold),
        "contamination": contamination,
        "expected_anomaly_count": int(math.ceil(n_rows * contamination)),
        "flagged_count": flagged_count,
        "flagged_fraction": round(float(flagged_count / n_rows), 4),
        "consensus": {
            "majority_detectors_required": majority_required,
            "flagged_count": consensus_count,
            "flagged_fraction": round(float(consensus_count / n_rows), 4),
        },
        "ensemble_summary": _summary(ensemble),
        "top_rows": top_rows,
        "projection_count": int(projections),
        "interpretation": (
            "Standard/deep anomaly screening combines robust feature deviations, sparse "
            "random-projection tail behaviour, and projection-density rarity. Research mode "
            "can confirm candidates with the heavier classical and neural ensembles."
        ),
    }
