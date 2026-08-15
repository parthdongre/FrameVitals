"""
Backward-compatible safe pandas imports.

Deprecated: use framevitals.safe_pandas instead.
"""

from framevitals.safe_pandas import (
    UnsafeExpression,
    safe_eval,
)

__all__ = [
    "UnsafeExpression",
    "safe_eval",
]
