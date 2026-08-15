"""
Backward-compatible profiling imports.

Deprecated: use framevitals.profiler instead.
"""

from framevitals.profiler import (
    build_profile,
    clean_value,
    detect_column_types,
    series_to_dict,
)

__all__ = [
    "build_profile",
    "clean_value",
    "detect_column_types",
    "series_to_dict",
]
