"""
Backward-compatible ML preprocessing imports.

Deprecated: use framevitals.ml_preprocessing instead.
"""

from framevitals.ml_preprocessing import (
    build_sklearn_preprocessor,
    get_transformed_feature_names,
    prepare_ml_matrix,
)

__all__ = [
    "build_sklearn_preprocessor",
    "get_transformed_feature_names",
    "prepare_ml_matrix",
]
