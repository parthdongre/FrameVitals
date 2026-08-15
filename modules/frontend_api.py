"""Backward-compatible frontend payload imports.

Deprecated: use framevitals.frontend_api instead.
"""

from framevitals.frontend_api import (
    build_dashboard_payload,
    format_bytes,
    format_elapsed_label,
)

__all__ = [
    "build_dashboard_payload",
    "format_bytes",
    "format_elapsed_label",
]
