"""
Backward-compatible feature importance imports.

Deprecated: use framevitals.feature_importance instead.
"""

from framevitals.feature_importance import (
    calculate_mutual_information,
    collapse_onehot_importances,
    run_feature_importance,
)

__all__ = [
    "calculate_mutual_information",
    "collapse_onehot_importances",
    "run_feature_importance",
]
