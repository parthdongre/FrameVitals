"""
Backward-compatible display signal imports.

Deprecated: use framevitals.signal_engine instead.
"""

from framevitals.signal_engine import (
    build_signals,
    severity_from_percent,
)

__all__ = [
    "build_signals",
    "severity_from_percent",
]
