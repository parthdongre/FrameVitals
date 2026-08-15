"""
Backward-compatible segment analysis imports.

Deprecated: use framevitals.segment_analysis instead.
"""

from framevitals.segment_analysis import (
    choose_segment_columns,
    run_segment_analysis,
)

__all__ = [
    "choose_segment_columns",
    "run_segment_analysis",
]
