"""Cheap column triage for expensive deep-analysis routines.

The profiler already describes every column. Deep statistical routines therefore
should not blindly spend bootstrap/distribution-fitting work on every feature.
This module scores all candidate columns with inexpensive vectorized statistics
and returns a stable, bounded subset for heavyweight diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_MODE_LIMITS: dict[str, tuple[int, int]] = {
    "quick": (4, 4),
    "standard": (8, 6),
    "deep": (12, 8),
    "research": (40, 20),
}


@dataclass(frozen=True, slots=True)
class DeepTriageResult:
    selected_numeric: tuple[str, ...]
    selected_categorical: tuple[str, ...]
    numeric_scores: dict[str, float]
    categorical_scores: dict[str, float]
    numeric_available: int
    categorical_available: int
    numeric_limit: int
    categorical_limit: int

    @property
    def selected_columns(self) -> tuple[str, ...]:
        return self.selected_numeric + self.selected_categorical

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "vectorized_interest_ranking",
            "selected_numeric": list(self.selected_numeric),
            "selected_categorical": list(self.selected_categorical),
            "numeric_available": self.numeric_available,
            "categorical_available": self.categorical_available,
            "numeric_limit": self.numeric_limit,
            "categorical_limit": self.categorical_limit,
            "numeric_truncated": self.numeric_available > len(self.selected_numeric),
            "categorical_truncated": self.categorical_available > len(self.selected_categorical),
            "numeric_scores": dict(self.numeric_scores),
            "categorical_scores": dict(self.categorical_scores),
        }


def _finite_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if np.isfinite(score) else 0.0


def _rank_scores(scores: dict[str, float], original_order: list[str], limit: int) -> tuple[str, ...]:
    order = {column: index for index, column in enumerate(original_order)}
    ranked = sorted(
        original_order,
        key=lambda column: (-_finite_score(scores.get(column)), order[column]),
    )
    return tuple(ranked[: max(0, min(limit, len(ranked)))])


def _numeric_interest_scores(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or frame.shape[1] == 0:
        return {}

    numeric = frame.replace([np.inf, -np.inf], np.nan)
    n_rows = max(len(numeric), 1)
    missing = numeric.isna().mean()
    unique = numeric.nunique(dropna=True)

    with np.errstate(all="ignore"):
        skew = numeric.skew(axis=0, skipna=True).abs().fillna(0.0)
        kurtosis = numeric.kurt(axis=0, skipna=True).abs().fillna(0.0)
        std = numeric.std(axis=0, skipna=True).fillna(0.0)

    # The score is intentionally diagnostic rather than predictive. High
    # missingness, skew/tails, near-constant behaviour and unusual cardinality
    # are the situations where bootstrap/distribution fitting adds most value.
    scores: dict[str, float] = {}
    for column in numeric.columns:
        unique_count = int(unique.get(column, 0))
        unique_ratio = unique_count / n_rows
        near_constant = 1.0 if unique_count <= 2 or _finite_score(std.get(column)) <= 1e-12 else 0.0
        score = (
            min(_finite_score(skew.get(column)), 8.0) * 1.5
            + min(_finite_score(kurtosis.get(column)), 20.0) * 0.65
            + min(_finite_score(missing.get(column)), 1.0) * 5.0
            + near_constant * 2.5
            + min(unique_ratio, 1.0) * 0.35
        )
        scores[str(column)] = round(float(score), 6)
    return scores


def _categorical_interest_scores(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or frame.shape[1] == 0:
        return {}

    n_rows = max(len(frame), 1)
    missing = frame.isna().mean()
    unique = frame.nunique(dropna=True)
    scores: dict[str, float] = {}

    for column in frame.columns:
        cardinality = int(unique.get(column, 0))
        missing_rate = min(_finite_score(missing.get(column)), 1.0)
        if cardinality <= 1:
            relationship_value = 0.5
        elif cardinality == 2:
            relationship_value = 3.0
        elif cardinality <= 12:
            relationship_value = 2.5
        elif cardinality <= 30:
            relationship_value = 1.5
        else:
            relationship_value = 0.25
        rarity_signal = min(cardinality / n_rows, 1.0) * 0.25
        score = missing_rate * 4.0 + relationship_value + rarity_signal
        scores[str(column)] = round(float(score), 6)
    return scores


def triage_deep_columns(dataframe: pd.DataFrame, *, mode: str) -> DeepTriageResult:
    """Rank all deep-statistics candidates and return a bounded stable subset."""
    if mode not in _MODE_LIMITS:
        raise ValueError(f"Unknown analysis mode: {mode}")

    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "bool", "string"],
    ).columns.tolist()
    numeric_limit, categorical_limit = _MODE_LIMITS[mode]

    numeric_scores = _numeric_interest_scores(dataframe[numeric_columns]) if numeric_columns else {}
    categorical_scores = (
        _categorical_interest_scores(dataframe[categorical_columns])
        if categorical_columns
        else {}
    )

    selected_numeric = _rank_scores(numeric_scores, numeric_columns, numeric_limit)
    selected_categorical = _rank_scores(
        categorical_scores,
        categorical_columns,
        categorical_limit,
    )

    return DeepTriageResult(
        selected_numeric=selected_numeric,
        selected_categorical=selected_categorical,
        numeric_scores=numeric_scores,
        categorical_scores=categorical_scores,
        numeric_available=len(numeric_columns),
        categorical_available=len(categorical_columns),
        numeric_limit=numeric_limit,
        categorical_limit=categorical_limit,
    )
