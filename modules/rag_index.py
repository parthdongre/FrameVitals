"""
Backward-compatible RAG index imports.

Deprecated: use framevitals.rag_index instead.
"""

from framevitals.rag_index import (
    Fact,
    build_fact_index,
    render_facts_block,
    retrieve,
)

__all__ = [
    "Fact",
    "build_fact_index",
    "render_facts_block",
    "retrieve",
]
