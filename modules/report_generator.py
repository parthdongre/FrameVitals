"""Backward-compatible report generator imports.

Deprecated: use framevitals.report_generator instead.
"""

from framevitals.report_generator import (
    REPORT_DIR,
    generate_pdf_report,
)

__all__ = [
    "REPORT_DIR",
    "generate_pdf_report",
]
