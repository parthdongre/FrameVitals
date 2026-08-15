"""
Backward-compatible model diagnostics imports.

Deprecated: use framevitals.model_diagnostics instead.
"""

from framevitals.model_diagnostics import (
    run_classification_diagnostics,
    run_model_diagnostics,
    run_regression_diagnostics,
)

__all__ = [
    "run_classification_diagnostics",
    "run_model_diagnostics",
    "run_regression_diagnostics",
]
