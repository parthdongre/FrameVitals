"""
Backward-compatible column role imports.

Deprecated: use framevitals.column_roles instead.
"""

from framevitals.column_roles import (
    get_columns_with_role,
    get_meaningful_categorical_columns,
    get_meaningful_numeric_columns,
    infer_column_roles,
    summarize_roles,
)

__all__ = [
    "get_columns_with_role",
    "get_meaningful_categorical_columns",
    "get_meaningful_numeric_columns",
    "infer_column_roles",
    "summarize_roles",
]
