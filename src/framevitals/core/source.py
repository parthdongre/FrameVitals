"""Recognition of structured sources without importing heavyweight frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SourceKind(str, Enum):
    TABULAR = "tabular"
    NESTED = "nested"
    GRAPH = "graph"
    TENSOR = "tensor"
    MODEL = "model"
    RELATIONAL = "relational"
    DOCUMENT = "document"
    STREAM = "stream"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceDescriptor:
    kind: SourceKind
    python_type: str
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["capabilities"] = list(self.capabilities)
        return payload


def _type_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_graph_like(value: Any) -> bool:
    module = type(value).__module__
    return (
        module.startswith("networkx.")
        and callable(getattr(value, "number_of_nodes", None))
        and callable(getattr(value, "number_of_edges", None))
        and hasattr(value, "nodes")
        and hasattr(value, "edges")
    )


def _is_torch_model(value: Any) -> bool:
    # User-defined nn.Module subclasses normally live in __main__ or an
    # application package rather than a torch.* module, so use the stable
    # nn.Module protocol instead of the class module path.
    return (
        callable(getattr(value, "state_dict", None))
        and callable(getattr(value, "named_parameters", None))
        and callable(getattr(value, "named_modules", None))
        and hasattr(value, "training")
    )


def _is_torch_tensor(value: Any) -> bool:
    module = type(value).__module__
    return (
        module.startswith("torch")
        and hasattr(value, "shape")
        and callable(getattr(value, "numel", None))
        and callable(getattr(value, "detach", None))
    )


def _is_tabular_object(value: Any) -> bool:
    module = type(value).__module__
    if module.startswith(("pandas.", "polars.", "pyarrow.")):
        return True

    # Fall back to stable dataframe/table capabilities instead of relying only
    # on implementation module paths, which can differ across versions.
    if hasattr(value, "shape") and hasattr(value, "columns"):
        return True
    if (
        hasattr(value, "schema")
        and hasattr(value, "column_names")
        and hasattr(value, "num_rows")
    ):
        return True
    return False


def _is_relational_mapping(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(_is_tabular_object(item) for item in value.values())
    )


def recognize_source(value: Any) -> SourceDescriptor:
    """Describe the broad structured source kind for protocol dispatch.

    Recognition is intentionally lightweight. Optional libraries such as
    NetworkX and PyTorch are never imported merely to inspect an object.
    """
    if _is_torch_model(value):
        return SourceDescriptor(
            kind=SourceKind.MODEL,
            python_type=_type_name(value),
            capabilities=(
                "parameters",
                "module_graph",
                "weights",
                "runtime_hooks",
                "gradients",
            ),
            metadata={"framework": "pytorch"},
        )

    if _is_graph_like(value):
        directed = bool(value.is_directed()) if callable(getattr(value, "is_directed", None)) else None
        multigraph = bool(value.is_multigraph()) if callable(getattr(value, "is_multigraph", None)) else None
        return SourceDescriptor(
            kind=SourceKind.GRAPH,
            python_type=_type_name(value),
            capabilities=(
                "topology",
                "connectivity",
                "centrality",
                "communities",
                "shortest_paths",
                "attributes",
            ),
            metadata={"directed": directed, "multigraph": multigraph},
        )

    if _is_torch_tensor(value):
        return SourceDescriptor(
            kind=SourceKind.TENSOR,
            python_type=_type_name(value),
            capabilities=("shape", "distribution", "rank", "sparsity"),
            metadata={"framework": "pytorch"},
        )

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return SourceDescriptor(
                kind=SourceKind.TENSOR,
                python_type=_type_name(value),
                capabilities=("shape", "distribution", "rank", "sparsity"),
                metadata={"framework": "numpy"},
            )
    except Exception:
        pass

    module = type(value).__module__
    if module.startswith(("pandas.", "polars.", "pyarrow.")):
        return SourceDescriptor(
            kind=SourceKind.TABULAR,
            python_type=_type_name(value),
            capabilities=("columns", "rows", "statistics", "relationships"),
        )

    if isinstance(value, (str, Path)):
        suffix = Path(value).suffix.lower()
        from framevitals.file_formats import format_for

        file_spec = format_for(value)
        if file_spec is not None and file_spec.category == "Document":
            return SourceDescriptor(
                kind=SourceKind.DOCUMENT,
                python_type=_type_name(value),
                capabilities=("text_extraction", "document_structure", "quality", "change"),
                metadata={"format": suffix.lstrip("."), "category": "document"},
            )
        if file_spec is not None and file_spec.category == "Tensor":
            return SourceDescriptor(
                kind=SourceKind.TENSOR,
                python_type=_type_name(value),
                capabilities=("shape", "distribution", "rank", "sparsity"),
                metadata={"format": suffix.lstrip("."), "framework": "numpy"},
            )
        if file_spec is not None and file_spec.category == "Nested":
            return SourceDescriptor(
                kind=SourceKind.NESTED,
                python_type=_type_name(value),
                capabilities=("nested", "schema", "type_conflicts"),
                metadata={"format": suffix.lstrip(".")},
            )
        if suffix == ".safetensors":
            return SourceDescriptor(
                kind=SourceKind.MODEL,
                python_type=_type_name(value),
                capabilities=("checkpoint_metadata", "parameters", "weights"),
                metadata={"framework": "safetensors", "format": "safetensors"},
            )
        if suffix == ".onnx":
            return SourceDescriptor(
                kind=SourceKind.MODEL,
                python_type=_type_name(value),
                capabilities=("model_graph", "parameters", "initializers", "operators"),
                metadata={"framework": "onnx", "format": "onnx"},
            )
        if suffix in {".graphml", ".gexf", ".gml"}:
            return SourceDescriptor(
                kind=SourceKind.GRAPH,
                python_type=_type_name(value),
                capabilities=(
                    "file",
                    "topology",
                    "connectivity",
                    "centrality",
                    "communities",
                    "shortest_paths",
                    "attributes",
                ),
                metadata={"format": suffix.lstrip(".")},
            )
        return SourceDescriptor(
            kind=SourceKind.TABULAR,
            python_type=_type_name(value),
            capabilities=("file", "columns", "rows"),
        )

    if _is_relational_mapping(value):
        return SourceDescriptor(
            kind=SourceKind.RELATIONAL,
            python_type=_type_name(value),
            capabilities=(
                "tables",
                "keys",
                "relationships",
                "referential_integrity",
                "join_risk",
            ),
            metadata={"tables": len(value)},
        )

    if isinstance(value, (dict, list, tuple)):
        return SourceDescriptor(
            kind=SourceKind.NESTED,
            python_type=_type_name(value),
            capabilities=("nested",),
        )

    return SourceDescriptor(
        kind=SourceKind.UNKNOWN,
        python_type=_type_name(value),
    )
