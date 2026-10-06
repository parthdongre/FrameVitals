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
    optimizer: Any = None,
) -> AnalysisResult | None:
    """Analyze supported non-tabular sources, returning None for tabular input."""
    descriptor = recognize_source(data)

    runtime_requested = (
        sample_batch is not None
        or targets is not None
        or loss_fn is not None
        or bool(backward)
        or max_runtime_modules is not None
        or optimizer is not None
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

    if descriptor.kind is SourceKind.NESTED:
        if runtime_requested:
            raise ValueError("Runtime model options are only valid for model Prism input.")
        from framevitals.analysis.nested import analyze_nested

        return analyze_nested(data, depth=depth)

    if descriptor.kind is SourceKind.RELATIONAL:
        if runtime_requested:
            raise ValueError("Runtime model options are only valid for model Prism input.")
        from framevitals.analysis.relational import analyze_relational

        return analyze_relational(data, depth=depth)

    if descriptor.kind is SourceKind.MODEL:
        model_format = descriptor.metadata.get("format")
        if model_format == "safetensors":
            if runtime_requested:
                raise ValueError(
                    "Runtime model options require an in-memory model, not a Safetensors file."
                )
            from framevitals.analysis.safetensors import analyze_safetensors

            return analyze_safetensors(data, depth=depth)

        if model_format == "onnx":
            if runtime_requested:
                raise ValueError(
                    "Runtime model options require an in-memory PyTorch model, not an ONNX file."
                )
            from framevitals.analysis.onnx_model import analyze_onnx

            return analyze_onnx(data, depth=depth)

        from framevitals.analysis.model import analyze_model

        return analyze_model(
            data,
            depth=depth,
            sample_batch=sample_batch,
            targets=targets,
            loss_fn=loss_fn,
            backward=backward,
            max_runtime_modules=max_runtime_modules,
            optimizer=optimizer,
        )

    return None



def compare_structured(reference: Any, current: Any):
    """Compare supported non-tabular sources, returning None for tabular pairs."""
    reference_descriptor = recognize_source(reference)
    current_descriptor = recognize_source(current)

    structured_kinds = {
        SourceKind.GRAPH,
        SourceKind.TENSOR,
        SourceKind.NESTED,
        SourceKind.RELATIONAL,
        SourceKind.MODEL,
    }
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
    if reference_descriptor.kind is SourceKind.NESTED:
        from framevitals.analysis.nested import compare_nested

        return compare_nested(reference, current)
    if reference_descriptor.kind is SourceKind.RELATIONAL:
        from framevitals.analysis.relational import compare_relational

        return compare_relational(reference, current)
    if reference_descriptor.kind is SourceKind.MODEL:
        reference_format = reference_descriptor.metadata.get("format")
        current_format = current_descriptor.metadata.get("format")

        if reference_format or current_format:
            if reference_format != current_format:
                raise TypeError(
                    "Model Tide requires matching model source formats when comparing files."
                )
            if reference_format == "safetensors":
                from framevitals.analysis.safetensors import compare_safetensors

                return compare_safetensors(reference, current)
            if reference_format == "onnx":
                from framevitals.analysis.onnx_model import compare_onnx

                return compare_onnx(reference, current)

        return compare_models(reference, current)

    return None
