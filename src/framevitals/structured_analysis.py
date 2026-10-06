"""Protocol-level dispatcher for non-tabular structured sources."""

from __future__ import annotations

from typing import Any

from framevitals.core.source import SourceKind, recognize_source
from framevitals.result import AnalysisResult


def analyze_structured(
    data: Any,
    *,
    depth: str | None = None,
    sample_batch: Any = None,
    targets: Any = None,
    loss_fn: Any = None,
    backward: bool = False,
    max_runtime_modules: int | None = None,
) -> AnalysisResult | None:
    """Analyze supported non-tabular sources, returning None for tabular input."""
    descriptor = recognize_source(data)

    runtime_requested = (
        sample_batch is not None
        or targets is not None
        or loss_fn is not None
        or bool(backward)
        or max_runtime_modules is not None
    )

    if descriptor.kind is SourceKind.GRAPH:
        if runtime_requested:
            raise ValueError("Runtime model options are only valid for model Prism input.")
        from framevitals.analysis.graph import analyze_graph

        return analyze_graph(data, depth=depth)

    if descriptor.kind is SourceKind.TENSOR:
        if runtime_requested:
            raise ValueError("Runtime model options are only valid for model Prism input.")
        from framevitals.analysis.tensor import analyze_tensor

        return analyze_tensor(data, depth=depth)

    if descriptor.kind is SourceKind.MODEL:
        from framevitals.analysis.model import analyze_model

        return analyze_model(
            data,
            depth=depth,
            sample_batch=sample_batch,
            targets=targets,
            loss_fn=loss_fn,
            backward=backward,
            max_runtime_modules=max_runtime_modules,
        )

    return None
