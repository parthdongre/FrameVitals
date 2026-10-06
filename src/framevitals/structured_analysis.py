"""Protocol-level dispatcher for non-tabular structured sources."""

from __future__ import annotations

from typing import Any

from framevitals.core.source import SourceKind, recognize_source
from framevitals.result import AnalysisResult


def analyze_structured(
    data: Any,
    *,
    depth: str | None = None,
) -> AnalysisResult | None:
    """Analyze supported non-tabular sources, returning None for tabular input."""
    descriptor = recognize_source(data)

    if descriptor.kind is SourceKind.GRAPH:
        from framevitals.analysis.graph import analyze_graph

        return analyze_graph(data, depth=depth)

    if descriptor.kind is SourceKind.TENSOR:
        from framevitals.analysis.tensor import analyze_tensor

        return analyze_tensor(data, depth=depth)

    if descriptor.kind is SourceKind.MODEL:
        from framevitals.analysis.model import analyze_model

        return analyze_model(data, depth=depth)

    return None
