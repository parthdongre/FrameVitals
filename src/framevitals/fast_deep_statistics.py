"""Fast deep-statistics execution for non-research analysis modes.

The public ``deep_statistics_v2`` implementation keeps its BCa bootstrap
semantics.  Budgeted quick/standard/deep execution can use this module to avoid
thousands of resamples per numeric column while preserving the rest of the v2
diagnostic battery.

Mean confidence intervals use the exact Student-t construction and median
confidence intervals use distribution-free order statistics derived from the
Binomial(n, 0.5) model.  Both are deterministic and O(n).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from framevitals.deep_statistics_v2 import (
    _categorical_column_stats,
    _classify_kurtosis,
    _classify_skew,
    _fit_best_distribution,
    _group_difference_test,
    _normality,
    _numeric_pair_stats,
    _outlier_flags,
    _point_biserial,
    _safe_float,
    run_deep_statistics_v2,
)


def _finite_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(
        dtype="float64",
        na_value=np.nan,
    )
    return values[np.isfinite(values)]


def fast_mean_ci(
    series: pd.Series,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Student-t confidence interval for the arithmetic mean in O(n)."""
    values = _finite_numeric(series)
    n = int(values.size)
    if n < 20:
        return {"available": False, "reason": "n<20"}

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if not math.isfinite(std):
        return {"available": False, "reason": "non-finite sample variance"}
    if std <= 1.0e-15:
        return {
            "available": True,
            "low": _safe_float(mean),
            "high": _safe_float(mean),
            "method": "student_t",
            "n_resamples": 0,
        }

    alpha = 1.0 - float(confidence)
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    half_width = critical * std / math.sqrt(n)
    return {
        "available": True,
        "low": _safe_float(mean - half_width),
        "high": _safe_float(mean + half_width),
        "method": "student_t",
        "n_resamples": 0,
    }


def fast_median_ci(
    series: pd.Series,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Distribution-free median CI via two selected order statistics.

    ``np.partition`` selects only the two required ranks, so no full sort or
    resampling matrix is allocated.
    """
    values = _finite_numeric(series)
    n = int(values.size)
    if n < 20:
        return {"available": False, "reason": "n<20"}

    alpha = 1.0 - float(confidence)
    rank = int(stats.binom.ppf(alpha / 2.0, n, 0.5))
    rank = max(1, min(rank, n // 2))

    low_index = rank - 1
    high_index = n - rank
    partitioned = np.partition(values, (low_index, high_index))

    return {
        "available": True,
        "low": _safe_float(partitioned[low_index]),
        "high": _safe_float(partitioned[high_index]),
        "method": "distribution_free_order_statistic",
        "n_resamples": 0,
        "rank_low": int(rank),
        "rank_high": int(n - rank + 1),
    }


def _numeric_column_stats_fast(series: pd.Series) -> dict[str, Any]:
    s = series.dropna()
    if s.empty:
        return {"status": "empty"}

    skew = _safe_float(s.skew())
    kurt = _safe_float(s.kurtosis())
    q1, q3 = _safe_float(s.quantile(0.25)), _safe_float(s.quantile(0.75))
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None

    return {
        "count": int(s.count()),
        "mean": _safe_float(s.mean()),
        "median": _safe_float(s.median()),
        "std": _safe_float(s.std()),
        "min": _safe_float(s.min()),
        "max": _safe_float(s.max()),
        "q1": q1,
        "q3": q3,
        "iqr": _safe_float(iqr),
        "skewness": skew,
        "skewness_label": _classify_skew(skew),
        "kurtosis": kurt,
        "kurtosis_label": _classify_kurtosis(kurt),
        "outliers": _outlier_flags(series),
        "normality": _normality(series),
        "distribution_fit": _fit_best_distribution(series),
        "bootstrap_mean_ci": fast_mean_ci(series),
        "bootstrap_median_ci": fast_median_ci(series),
    }


def _numeric_bivariate(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    max_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    numeric_pairs: list[dict[str, Any]] = []
    budget = max_pairs
    for index, left in enumerate(numeric_cols):
        for right in numeric_cols[index + 1 :]:
            if budget <= 0:
                break
            payload = _numeric_pair_stats(df[left], df[right])
            if payload.get("available"):
                numeric_pairs.append(
                    {"column_a": left, "column_b": right, **payload}
                )
                budget -= 1
        if budget <= 0:
            break
    numeric_pairs.sort(
        key=lambda item: abs(item.get("pearson", {}).get("r") or 0),
        reverse=True,
    )

    binary_numeric_pairs: list[dict[str, Any]] = []
    budget = max_pairs
    for category in categorical_cols:
        if df[category].nunique(dropna=True) != 2:
            continue
        for numeric in numeric_cols:
            if budget <= 0:
                break
            payload = _point_biserial(df[category], df[numeric])
            if payload.get("available"):
                binary_numeric_pairs.append(
                    {
                        "binary_column": category,
                        "numeric_column": numeric,
                        **payload,
                    }
                )
                budget -= 1
        if budget <= 0:
            break

    group_difference_tests: list[dict[str, Any]] = []
    budget = max_pairs
    for category in categorical_cols:
        unique = df[category].nunique(dropna=True)
        if unique < 2 or unique > 6:
            continue
        for numeric in numeric_cols:
            if budget <= 0:
                break
            payload = _group_difference_test(df[numeric], df[category])
            if payload.get("available"):
                group_difference_tests.append(
                    {
                        "group_column": category,
                        "numeric_column": numeric,
                        **payload,
                    }
                )
                budget -= 1
        if budget <= 0:
            break

    return numeric_pairs, binary_numeric_pairs, group_difference_tests


def run_fast_deep_statistics_v2(
    df: pd.DataFrame,
    max_pairs: int = 20,
) -> dict[str, Any]:
    """Run the v2 diagnostic battery with O(n) confidence intervals."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    categorical_view = (
        df.loc[:, categorical_cols]
        if categorical_cols
        else pd.DataFrame(index=df.index)
    )
    result = run_deep_statistics_v2(categorical_view, max_pairs=max_pairs)

    result["numeric_columns"] = numeric_cols
    result["numeric_statistics"] = {
        column: _numeric_column_stats_fast(df[column])
        for column in numeric_cols
    }

    (
        numeric_pairs,
        binary_numeric_pairs,
        group_difference_tests,
    ) = _numeric_bivariate(
        df,
        numeric_cols,
        categorical_cols,
        max_pairs=max_pairs,
    )

    result["bivariate"]["numeric_pairs"] = numeric_pairs
    result["bivariate"]["binary_numeric_pairs"] = binary_numeric_pairs
    result["bivariate"]["group_difference_tests"] = group_difference_tests

    result["summary"].update(
        {
            "numeric_count": len(numeric_cols),
            "numeric_pairs_tested": len(numeric_pairs),
            "binary_numeric_pairs_tested": len(binary_numeric_pairs),
            "group_difference_tests_run": len(group_difference_tests),
        }
    )
    result["inference"] = {
        "method": "fast_closed_form_and_order_statistics",
        "mean_ci": "student_t",
        "median_ci": "distribution_free_order_statistic",
        "bootstrap_resamples": 0,
    }
    return result
