"""
Backward-compatible target leakage imports.

Deprecated: use framevitals.target_leakage instead.
"""

from framevitals.target_leakage import (
    classify_target_leakage_risk,
    numeric_correlation,
    run_target_leakage_analysis,
    same_ratio_non_missing,
)

__all__ = [
    "classify_target_leakage_risk",
    "numeric_correlation",
    "run_target_leakage_analysis",
    "same_ratio_non_missing",
]
