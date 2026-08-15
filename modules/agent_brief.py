"""
Backward-compatible dataset brief imports.

Deprecated: use framevitals.agent_brief instead.
"""

from framevitals.agent_brief import (
    build_dataset_brief,
    render_brief_block,
)

__all__ = [
    "build_dataset_brief",
    "render_brief_block",
]
