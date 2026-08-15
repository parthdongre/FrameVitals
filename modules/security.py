"""
Backward-compatible security imports.

Deprecated: use framevitals.security instead.
"""

from framevitals.security import (
    ALLOWED_EXTENSIONS,
    make_safe_filename,
    sanitize_csv_value,
    validate_file,
)


__all__ = [
    "ALLOWED_EXTENSIONS",
    "make_safe_filename",
    "sanitize_csv_value",
    "validate_file",
]
