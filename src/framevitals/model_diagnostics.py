"""Compatibility shim for legacy tabular predictive diagnostics.

The real implementation moved to framevitals.predictive_diagnostics so the
model_diagnostics name can refer unambiguously to actual model/checkpoint
inspection in the new structured architecture.
"""

from framevitals.predictive_diagnostics import (
    run_classification_diagnostics,
    run_model_diagnostics,
    run_predictive_diagnostics,
    run_regression_diagnostics,
)

__all__ = [
    "run_classification_diagnostics",
    "run_regression_diagnostics",
    "run_predictive_diagnostics",
    "run_model_diagnostics",
]
