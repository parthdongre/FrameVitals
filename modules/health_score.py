"""
Backward-compatible health score imports.

Deprecated: use framevitals.health_score instead.
"""

from framevitals.health_score import (
    calculate_health_score,
    calculate_outlier_percent,
)

__all__ = [
    "calculate_health_score",
    "calculate_outlier_percent",
]
