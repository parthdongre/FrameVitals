"""
Backward-compatible time-series imports.

Deprecated: use framevitals.time_series instead.
"""

from framevitals.time_series import (
    detect_and_analyze_time_series,
    detect_date_column,
)

__all__ = [
    "detect_and_analyze_time_series",
    "detect_date_column",
]
