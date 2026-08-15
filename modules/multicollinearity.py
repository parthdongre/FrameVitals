"""
Backward-compatible multicollinearity imports.

Deprecated: use framevitals.multicollinearity instead.
"""

from framevitals.multicollinearity import (
    calculate_vif_scores,
    detect_redundant_feature_groups,
    prepare_numeric_matrix,
    run_multicollinearity_analysis,
)

__all__ = [
    "calculate_vif_scores",
    "detect_redundant_feature_groups",
    "prepare_numeric_matrix",
    "run_multicollinearity_analysis",
]
