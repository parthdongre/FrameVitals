"""
Backward-compatible drift analysis imports.

Deprecated: use framevitals.drift_analysis instead.
"""

from framevitals.drift_analysis import (
    compare_datasets,
    split_by_date,
)

__all__ = [
    "compare_datasets",
    "split_by_date",
]
