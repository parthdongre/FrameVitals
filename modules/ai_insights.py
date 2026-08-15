"""
Backward-compatible AI insight imports.

Deprecated: use framevitals.ai_insights instead.
"""

from framevitals.ai_insights import (
    answer_dataset_question,
    compact_context,
    generate_ai_report,
)

__all__ = [
    "answer_dataset_question",
    "compact_context",
    "generate_ai_report",
]
