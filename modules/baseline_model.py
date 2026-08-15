"""
Backward-compatible baseline model imports.

Deprecated: use framevitals.baseline_model instead.
"""

from framevitals.baseline_model import (
    classification_metrics,
    regression_metrics,
    run_baseline_model,
)

__all__ = [
    "classification_metrics",
    "regression_metrics",
    "run_baseline_model",
]
