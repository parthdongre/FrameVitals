"""
Backward-compatible visualization imports.

Deprecated: use framevitals.visualizer instead.
"""

from framevitals.visualizer import (
    CHART_DIR,
    generate_charts,
)

__all__ = [
    "CHART_DIR",
    "generate_charts",
]
