"""
Backward-compatible cleaning imports.

Deprecated: use framevitals.cleaner instead.
"""

from framevitals.cleaner import (
    CLEANED_DIR,
    create_cleaned_dataset,
)

__all__ = [
    "CLEANED_DIR",
    "create_cleaned_dataset",
]
