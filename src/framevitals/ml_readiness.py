"""ML-readiness scoring and compatibility helpers."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np


_CATEGORICAL_DTYPES = ["object", "string", "category", "bool"]


def _score_ml_readiness(
    *,
    rows: int,
    columns: int,
    missing_total: int,
    duplicate_percent: float,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> dict[str, Any]:
    missing_percent = float(missing_total / max(rows * columns, 1) * 100)

    encoding_penalty = min(20, len(categorical_cols) * 2)
    missing_penalty = min(30, missing_percent)
    duplicate_penalty = min(15, duplicate_percent)

    score = 100 - missing_penalty - duplicate_penalty - encoding_penalty
    score = round(max(0, min(100, score)), 2)

    if score >= 85:
        label = "Ready"
    elif score >= 70:
        label = "Mostly Ready"
    elif score >= 50:
        label = "Partially Ready"
    else:
        label = "Not Ready"

    return {
        "score": score,
        "label": label,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "issues": {
            "missing_percent": round(missing_percent, 2),
            "duplicate_percent": round(duplicate_percent, 2),
            "encoding_required_count": len(categorical_cols),
        },
        "recommendations": [
            "Handle missing values before model training.",
            "Encode categorical columns.",
            "Remove duplicates if they are not valid repeated records.",
            "Select a clear target column for supervised learning.",
        ],
    }


def calculate_ml_readiness_from_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Calculate ML-readiness directly from a completed dataset profile."""
    shape = profile.get("shape", {})
    rows = int(shape.get("rows", 0) or 0)
    columns = int(shape.get("columns", len(profile.get("columns", []))) or 0)
    missing_total = sum(
        int(value)
        for value in profile.get("missing_counts", {}).values()
        if value is not None
    )
    duplicate_percent = float(profile.get("duplicate_percent", 0.0))
    categorical_cols = list(profile.get("categorical_columns", []))
    numeric_cols = list(profile.get("numeric_columns", []))

    return _score_ml_readiness(
        rows=rows,
        columns=columns,
        missing_total=missing_total,
        duplicate_percent=duplicate_percent,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )


def calculate_ml_readiness(df, profile=None):
    """Calculate ML-readiness while reusing profile metrics when available.

    ``profile`` is optional to preserve the standalone helper API. The main
    pipeline passes the already-built profile so FrameVitals does not rescan
    the full dataset for missing values, duplicates, and basic column groups.
    """
    rows, columns = df.shape

    if profile is None:
        # Standalone compatibility path. The public focused API builds a profile
        # first and therefore avoids repeating these scans.
        missing_total = sum(int(df[column].isna().sum()) for column in df.columns)
        duplicate_percent = float(df.duplicated().sum() / max(rows, 1) * 100)
        categorical_cols = df.select_dtypes(include=_CATEGORICAL_DTYPES).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        missing_total = sum(
            int(value)
            for value in profile.get("missing_counts", {}).values()
            if value is not None
        )
        duplicate_percent = float(profile.get("duplicate_percent", 0.0))
        categorical_cols = list(profile.get("categorical_columns", []))
        numeric_cols = list(profile.get("numeric_columns", []))

    return _score_ml_readiness(
        rows=int(rows),
        columns=int(columns),
        missing_total=missing_total,
        duplicate_percent=duplicate_percent,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
    )


class _CallableReadinessModule(ModuleType):
    """Preserve ``fv.ml_readiness(data)`` after importing this compatibility module.

    Python normally places an imported submodule on its parent package under the
    same attribute name. Since FrameVitals historically also exposes the focused
    function ``framevitals.ml_readiness(...)``, importing this module would
    otherwise replace that function with a non-callable module object. Making
    the compatibility module callable preserves both APIs without eager imports.
    """

    def __call__(self, data: Any) -> dict[str, Any]:
        from framevitals.focused import ml_readiness as public_ml_readiness

        return public_ml_readiness(data)


sys.modules[__name__].__class__ = _CallableReadinessModule
