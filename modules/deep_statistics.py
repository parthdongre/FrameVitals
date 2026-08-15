"""
Backward-compatible deep statistics imports.

Deprecated: use framevitals.deep_statistics instead.
"""

from framevitals.deep_statistics import (
    categorical_deep_statistics,
    chi_square_relationships,
    classify_correlation,
    classify_kurtosis,
    classify_skewness,
    correlation_insights,
    normality_test,
    numeric_deep_statistics,
    run_deep_statistics,
    safe_float,
)

__all__ = [
    "categorical_deep_statistics",
    "chi_square_relationships",
    "classify_correlation",
    "classify_kurtosis",
    "classify_skewness",
    "correlation_insights",
    "normality_test",
    "numeric_deep_statistics",
    "run_deep_statistics",
    "safe_float",
]
