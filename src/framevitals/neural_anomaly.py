"""Small neural reconstruction detector for research-mode anomaly analysis.

This intentionally uses the existing scikit-learn dependency instead of adding a
large deep-learning runtime. The network is bounded by rows, columns and epochs
and is designed as an additional nonlinear anomaly view, not as a replacement
for deterministic statistical checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _unit_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def neural_reconstruction_anomalies(
    dataframe: pd.DataFrame,
    *,
    max_rows: int = 3_000,
    max_columns: int = 24,
    max_iter: int = 35,
    top_k: int = 25,
    random_state: int = 42,
) -> dict[str, Any]:
    """Score rows with a tiny autoencoder-like MLP reconstruction network."""
    if max_rows < 20:
        raise ValueError("max_rows must be at least 20.")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")
    if max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    numeric = dataframe.select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    dropped_constant: list[str] = []
    for column in list(numeric.columns):
        if numeric[column].nunique(dropna=True) <= 1:
            numeric = numeric.drop(columns=[column])
            dropped_constant.append(str(column))

    if numeric.shape[1] < 2 or len(numeric) < 20:
        return {
            "available": False,
            "reason": "Need at least 20 rows and two non-constant numeric columns.",
            "dropped_constant_columns": dropped_constant,
        }

    variances = numeric.var(axis=0, skipna=True).fillna(0.0)
    selected = variances.sort_values(ascending=False).head(max_columns).index.tolist()
    numeric = numeric[selected]

    source_rows = len(numeric)
    if source_rows > max_rows:
        positions = np.linspace(0, source_rows - 1, num=max_rows, dtype=np.int64)
        positions = np.unique(positions)
        work = numeric.iloc[positions]
        sampled = True
    else:
        work = numeric
        sampled = False

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(work)
    matrix = scaler.fit_transform(matrix)

    n_features = matrix.shape[1]
    bottleneck = max(2, min(12, n_features // 3 if n_features >= 6 else n_features - 1))
    hidden = max(bottleneck + 1, min(24, max(4, n_features // 2)))
    network = MLPRegressor(
        hidden_layer_sizes=(hidden, bottleneck, hidden),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(256, max(16, len(matrix) // 10)),
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=random_state,
    )
    network.fit(matrix, matrix)
    reconstructed = network.predict(matrix)
    errors = np.mean((matrix - reconstructed) ** 2, axis=1)
    scores = _unit_scale(errors)

    order = np.argsort(scores)[::-1][: min(top_k, len(scores))]
    top_rows = [
        {
            "row_index": str(work.index[int(index)]),
            "score": round(float(scores[int(index)]), 6),
            "reconstruction_error": round(float(errors[int(index)]), 6),
        }
        for index in order
    ]

    return {
        "available": True,
        "method": "bounded_mlp_reconstruction",
        "source_rows": int(source_rows),
        "sample_rows": int(len(work)),
        "sampled": sampled,
        "used_columns": [str(column) for column in selected],
        "columns_available": int(dataframe.select_dtypes(include=[np.number]).shape[1]),
        "columns_used": int(len(selected)),
        "truncated_columns": bool(len(selected) < dataframe.select_dtypes(include=[np.number]).shape[1]),
        "architecture": [int(n_features), int(hidden), int(bottleneck), int(hidden), int(n_features)],
        "iterations": int(network.n_iter_),
        "loss": round(float(network.loss_), 6),
        "score_summary": {
            "mean": round(float(np.mean(scores)), 6),
            "p95": round(float(np.quantile(scores, 0.95)), 6),
            "p99": round(float(np.quantile(scores, 0.99)), 6),
            "max": round(float(np.max(scores)), 6),
        },
        "top_rows": top_rows,
        "dropped_constant_columns": dropped_constant,
    }
