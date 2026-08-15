"""
Backward-compatible advanced indicator imports.

Deprecated: use framevitals.advanced_indicators instead.
"""

from framevitals.advanced_indicators import (
    calculate_advanced_indicators,
    calculate_anomalies,
    calculate_column_utility,
    calculate_freshness,
    detect_fairness_review,
)

__all__ = [
    "calculate_advanced_indicators",
    "calculate_anomalies",
    "calculate_column_utility",
    "calculate_freshness",
    "detect_fairness_review",
]
