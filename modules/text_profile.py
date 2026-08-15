"""
Backward-compatible text profiling imports.

Deprecated: use framevitals.text_profile instead.
"""

from framevitals.text_profile import (
    detect_text_columns,
    profile_text_columns,
)

__all__ = [
    "detect_text_columns",
    "profile_text_columns",
]
