"""
Backward-compatible target analysis imports.

Deprecated: use framevitals.target_analyzer instead.
"""

from framevitals.target_analyzer import (
    analyze_classification_target,
    analyze_regression_target,
    analyze_target,
    detect_task_type,
)

__all__ = [
    "analyze_classification_target",
    "analyze_regression_target",
    "analyze_target",
    "detect_task_type",
]
