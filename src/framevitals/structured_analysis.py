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
        if descriptor.metadata.get("format") == "safetensors":
            if runtime_requested:
                raise ValueError(
                    "Runtime model options require an in-memory model, not a Safetensors file."
                )
            from framevitals.analysis.safetensors import analyze_safetensors

            return analyze_safetensors(data, depth=depth)

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



def compare_structured(reference: Any, current: Any):
    """Compare supported non-tabular sources, returning None for tabular pairs."""
    reference_descriptor = recognize_source(reference)
    current_descriptor = recognize_source(current)

    structured_kinds = {SourceKind.GRAPH, SourceKind.TENSOR, SourceKind.MODEL}
    reference_structured = reference_descriptor.kind in structured_kinds
    current_structured = current_descriptor.kind in structured_kinds

    if not reference_structured and not current_structured:
        return None

    if reference_descriptor.kind is not current_descriptor.kind:
        raise TypeError(
            "Structured Tide requires reference and current inputs of the same source kind "
            f"(got {reference_descriptor.kind.value} and {current_descriptor.kind.value})."
        )

    from framevitals.analysis.change import (
        compare_graphs,
        compare_models,
        compare_tensors,
    )

    if reference_descriptor.kind is SourceKind.GRAPH:
        return compare_graphs(reference, current)
    if reference_descriptor.kind is SourceKind.TENSOR:
        return compare_tensors(reference, current)
    if reference_descriptor.kind is SourceKind.MODEL:
        if (
            reference_descriptor.metadata.get("format") == "safetensors"
            and current_descriptor.metadata.get("format") == "safetensors"
        ):
            from framevitals.analysis.safetensors import compare_safetensors

            return compare_safetensors(reference, current)
        return compare_models(reference, current)

    return None
