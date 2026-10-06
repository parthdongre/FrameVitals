"""Axiom contracts for non-tabular structured sources."""

from __future__ import annotations

from typing import Any

from framevitals.core.source import SourceKind, recognize_source
from framevitals.quality_results import ValidationResult


def _finding(
    code: str,
    message: str,
    *,
    severity: str = "error",
    field: str = "structure",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "column": field,
        "message": message,
    }


def _model_structure(source: Any, descriptor: Any) -> dict[str, Any]:
    model_format = descriptor.metadata.get("format")
    if model_format == "safetensors":
        from framevitals.analysis.safetensors import inspect_safetensors

        summary = inspect_safetensors(source)
        return {
            "framework": "safetensors",
            "format": "safetensors",
            "architecture": summary.get("architecture"),
            "parameters": {
                item["name"]: {
                    "shape": list(item["shape"]),
                    "dtype": item["dtype"],
                }
                for item in summary["tensors"]
            },
        }

    if model_format == "onnx":
        from framevitals.analysis.onnx_model import inspect_onnx

        summary = inspect_onnx(source)
        return {
            "framework": "onnx",
            "format": "onnx",
            "architecture": summary.get("architecture"),
            "operators": dict(summary.get("operator_counts", {})),
            "parameters": {
                item["name"]: {
                    "shape": list(item["shape"]),
                    "dtype": item["dtype"],
                }
                for item in summary["initializers"]
            },
        }

    parameters = {}
    for name, parameter in source.named_parameters():
        parameters[str(name)] = {
            "shape": [int(v) for v in parameter.shape],
            "dtype": str(parameter.dtype),
            "requires_grad": bool(getattr(parameter, "requires_grad", False)),
        }

    module_types: dict[str, int] = {}
    for _, module in source.named_modules():
        name = type(module).__name__
        module_types[name] = module_types.get(name, 0) + 1

    return {
        "framework": "pytorch",
        "format": "in_memory",
        "architecture": None,
        "class_name": type(source).__name__,
        "module_types": module_types,
        "parameters": parameters,
    }


def infer_structured_contract(
    reference: Any,
    *,
    tolerance: float = 0.10,
) -> dict[str, Any] | None:
    """Infer a compact structural contract for supported non-tabular sources."""
    descriptor = recognize_source(reference)
    kind = descriptor.kind

    if kind is SourceKind.GRAPH:
        from framevitals.analysis.graph import analyze_graph

        result = analyze_graph(reference, depth="quick")
        graph = result["graph"]
        components = graph.get("components", {})
        nodes = int(graph.get("nodes", 0))
        isolate_ratio = float(graph.get("isolates", 0)) / max(1, nodes)
        return {
            "contract_schema_version": "1",
            "source_kind": "graph",
            "expectations": {
                "directed": bool(graph.get("directed")),
                "multigraph": bool(graph.get("multigraph")),
                "min_nodes": max(0, int(nodes * (1.0 - tolerance))),
                "max_nodes": max(0, int(nodes * (1.0 + tolerance)) + 1),
                "max_isolate_ratio": min(1.0, isolate_ratio + tolerance),
                "min_largest_component_ratio": max(
                    0.0,
                    float(components.get("largest_component_ratio") or 0.0) - tolerance,
                ),
            },
        }

    if kind is SourceKind.TENSOR:
        from framevitals.analysis.tensor import tensor_metrics

        metrics = tensor_metrics(reference, sample_values=100_000)
        matrix = metrics.get("matrix") or {}
        expectations: dict[str, Any] = {
            "shape": list(metrics.get("shape", [])),
            "dtype": metrics.get("dtype"),
            "require_finite": (
                float(metrics.get("nan_fraction_sample") or 0.0) == 0.0
                and float(metrics.get("inf_fraction_sample") or 0.0) == 0.0
            ),
        }
        rank_ratio = matrix.get("rank_ratio") if isinstance(matrix, dict) else None
        if isinstance(rank_ratio, (int, float)):
            expectations["min_rank_ratio"] = max(
                0.0,
                float(rank_ratio) - tolerance,
            )
        return {
            "contract_schema_version": "1",
            "source_kind": "tensor",
            "expectations": expectations,
        }

    if kind is SourceKind.NESTED:
        from framevitals.analysis.nested import inspect_nested

        summary = inspect_nested(reference, max_nodes=100_000, max_depth=32)
        top_keys = sorted(str(key) for key in reference) if isinstance(reference, dict) else []
        return {
            "contract_schema_version": "1",
            "source_kind": "nested",
            "expectations": {
                "required_top_level_keys": top_keys,
                "max_depth": int(summary["max_depth"]) + 2,
                "max_type_conflicts": int(summary["path_type_conflict_count"]),
                "allow_cycles": int(summary["cyclic_references"]) > 0,
            },
        }

    if kind is SourceKind.RELATIONAL:
        from framevitals.analysis.relational import inspect_relational

        summary = inspect_relational(reference, max_rows_per_table=100_000)
        relationship_signatures = sorted(
            [
                [
                    item["left_table"],
                    item["right_table"],
                    item["column"],
                    item["cardinality"],
                ]
                for item in summary["relationships"]
            ]
        )
        return {
            "contract_schema_version": "1",
            "source_kind": "relational",
            "expectations": {
                "required_tables": sorted(summary["tables"]),
                "required_columns": {
                    name: sorted(table["column_names"])
                    for name, table in summary["tables"].items()
                },
                "key_candidates": {
                    name: sorted(table["key_candidates"])
                    for name, table in summary["tables"].items()
                },
                "relationship_signatures": relationship_signatures,
            },
        }

    if kind is SourceKind.MODEL:
        structure = _model_structure(reference, descriptor)
        return {
            "contract_schema_version": "1",
            "source_kind": "model",
            "expectations": structure,
        }

    return None


def validate_structured(
    current: Any,
    contract: dict[str, Any],
) -> ValidationResult:
    """Validate one structured source against a structured Axiom contract."""
    kind = str(contract.get("source_kind") or "")
    expectations = contract.get("expectations")
    if not isinstance(expectations, dict):
        raise ValueError("Structured contract is missing expectations.")

    descriptor = recognize_source(current)
    if descriptor.kind.value != kind:
        finding = _finding(
            "axiom.source_kind",
            f"Expected source kind {kind!r}, observed {descriptor.kind.value!r}.",
        )
        return ValidationResult({
            "valid": False,
            "status": "fail",
            "summary": {"errors": 1, "warnings": 0, "columns_checked": 0},
            "findings": [finding],
        })

    findings: list[dict[str, Any]] = []

    if kind == "graph":
        from framevitals.analysis.graph import analyze_graph

        graph = analyze_graph(current, depth="quick")["graph"]
        nodes = int(graph.get("nodes", 0))
        isolate_ratio = float(graph.get("isolates", 0)) / max(1, nodes)
        largest_ratio = float(
            (graph.get("components") or {}).get("largest_component_ratio") or 0.0
        )
        if bool(graph.get("directed")) != bool(expectations.get("directed")):
            findings.append(_finding(
                "axiom.graph.directed",
                "Graph directionality differs from the reference contract.",
            ))
        if bool(graph.get("multigraph")) != bool(expectations.get("multigraph")):
            findings.append(_finding(
                "axiom.graph.multigraph",
                "Graph multigraph semantics differ from the reference contract.",
            ))
        if nodes < int(expectations.get("min_nodes", 0)):
            findings.append(_finding(
                "axiom.graph.min_nodes",
                f"Graph has {nodes} nodes, below the expected minimum.",
            ))
        if nodes > int(expectations.get("max_nodes", nodes)):
            findings.append(_finding(
                "axiom.graph.max_nodes",
                f"Graph has {nodes} nodes, above the expected maximum.",
            ))
        if isolate_ratio > float(expectations.get("max_isolate_ratio", 1.0)):
            findings.append(_finding(
                "axiom.graph.isolates",
                f"Isolate ratio {isolate_ratio:.3f} exceeds the allowed bound.",
            ))
        if largest_ratio < float(expectations.get("min_largest_component_ratio", 0.0)):
            findings.append(_finding(
                "axiom.graph.connectivity",
                f"Largest-component ratio {largest_ratio:.3f} is below the required bound.",
            ))

    elif kind == "tensor":
        from framevitals.analysis.tensor import tensor_metrics

        metrics = tensor_metrics(current, sample_values=100_000)
        if list(metrics.get("shape", [])) != list(expectations.get("shape", [])):
            findings.append(_finding(
                "axiom.tensor.shape",
                f"Expected shape {expectations.get('shape')}, observed {metrics.get('shape')}.",
            ))
        if str(metrics.get("dtype")) != str(expectations.get("dtype")):
            findings.append(_finding(
                "axiom.tensor.dtype",
                f"Expected dtype {expectations.get('dtype')}, observed {metrics.get('dtype')}.",
            ))
        if expectations.get("require_finite"):
            if (
                float(metrics.get("nan_fraction_sample") or 0.0) > 0
                or float(metrics.get("inf_fraction_sample") or 0.0) > 0
            ):
                findings.append(_finding(
                    "axiom.tensor.finite",
                    "Tensor contains sampled NaN/Inf values.",
                ))
        matrix = metrics.get("matrix") or {}
        rank_ratio = matrix.get("rank_ratio") if isinstance(matrix, dict) else None
        min_rank = expectations.get("min_rank_ratio")
        if (
            isinstance(rank_ratio, (int, float))
            and isinstance(min_rank, (int, float))
            and float(rank_ratio) < float(min_rank)
        ):
            findings.append(_finding(
                "axiom.tensor.rank",
                f"Rank ratio {float(rank_ratio):.3f} is below {float(min_rank):.3f}.",
            ))

    elif kind == "nested":
        from framevitals.analysis.nested import inspect_nested

        summary = inspect_nested(current, max_nodes=100_000, max_depth=32)
        if isinstance(current, dict):
            missing = sorted(
                set(expectations.get("required_top_level_keys", [])) - set(map(str, current))
            )
            if missing:
                findings.append(_finding(
                    "axiom.nested.keys",
                    f"Missing required top-level keys: {missing}.",
                ))
        if int(summary["max_depth"]) > int(expectations.get("max_depth", 10**9)):
            findings.append(_finding(
                "axiom.nested.depth",
                f"Nested depth {summary['max_depth']} exceeds the allowed maximum.",
            ))
        if int(summary["path_type_conflict_count"]) > int(
            expectations.get("max_type_conflicts", 10**9)
        ):
            findings.append(_finding(
                "axiom.nested.type_conflicts",
                "Nested type conflicts increased beyond the reference contract.",
            ))
        if not expectations.get("allow_cycles") and int(summary["cyclic_references"]) > 0:
            findings.append(_finding(
                "axiom.nested.cycles",
                "Cyclic references are not allowed by the contract.",
            ))

    elif kind == "relational":
        from framevitals.analysis.relational import inspect_relational

        summary = inspect_relational(current, max_rows_per_table=100_000)
        tables = summary["tables"]
        missing_tables = sorted(
            set(expectations.get("required_tables", [])) - set(tables)
        )
        if missing_tables:
            findings.append(_finding(
                "axiom.relational.tables",
                f"Missing required tables: {missing_tables}.",
            ))
        for table, required_columns in expectations.get("required_columns", {}).items():
            if table not in tables:
                continue
            missing = sorted(
                set(required_columns) - set(tables[table]["column_names"])
            )
            if missing:
                findings.append(_finding(
                    "axiom.relational.columns",
                    f"Table {table!r} is missing required columns: {missing}.",
                    field=table,
                ))

        for table, required_keys in expectations.get("key_candidates", {}).items():
            if table not in tables:
                continue
            missing_keys = sorted(
                set(required_keys) - set(tables[table]["key_candidates"])
            )
            if missing_keys:
                findings.append(_finding(
                    "axiom.relational.keys",
                    f"Table {table!r} no longer preserves key uniqueness for: {missing_keys}.",
                    field=table,
                ))

        observed_relationships = {
            (
                item["left_table"],
                item["right_table"],
                item["column"],
                item["cardinality"],
            )
            for item in summary["relationships"]
        }
        required_relationships = {
            tuple(item)
            for item in expectations.get("relationship_signatures", [])
        }
        missing_relationships = sorted(required_relationships - observed_relationships)
        if missing_relationships:
            findings.append(_finding(
                "axiom.relational.relationships",
                f"{len(missing_relationships)} inferred relationships no longer match the contract.",
            ))

    elif kind == "model":
        observed = _model_structure(current, descriptor)

        expected_format = expectations.get("format")
        observed_format = observed.get("format")
        if expected_format != observed_format:
            findings.append(_finding(
                "axiom.model.format",
                f"Expected model format {expected_format!r}, observed {observed_format!r}.",
            ))

        expected_framework = expectations.get("framework")
        observed_framework = observed.get("framework")
        if expected_framework != observed_framework:
            findings.append(_finding(
                "axiom.model.framework",
                f"Expected framework {expected_framework!r}, observed {observed_framework!r}.",
            ))

        expected_architecture = expectations.get("architecture")
        observed_architecture = observed.get("architecture")
        if (
            expected_architecture
            and observed_architecture
            and expected_architecture != observed_architecture
        ):
            findings.append(_finding(
                "axiom.model.architecture",
                f"Expected architecture {expected_architecture!r}, observed {observed_architecture!r}.",
                severity="warning",
            ))

        expected_parameters = expectations.get("parameters", {})
        observed_parameters = observed.get("parameters", {})

        missing = sorted(set(expected_parameters) - set(observed_parameters))
        added = sorted(set(observed_parameters) - set(expected_parameters))
        if missing:
            findings.append(_finding(
                "axiom.model.parameters_missing",
                f"Missing model parameters: {missing[:20]}.",
            ))
        if added:
            findings.append(_finding(
                "axiom.model.parameters_added",
                f"Unexpected model parameters: {added[:20]}.",
                severity="warning",
            ))

        for name in sorted(set(expected_parameters) & set(observed_parameters)):
            expected = expected_parameters[name]
            actual = observed_parameters[name]
            if list(expected.get("shape", [])) != list(actual.get("shape", [])):
                findings.append(_finding(
                    "axiom.model.shape",
                    f"Parameter {name!r} changed shape from {expected.get('shape')} to {actual.get('shape')}.",
                    field=name,
                ))
            if str(expected.get("dtype")) != str(actual.get("dtype")):
                findings.append(_finding(
                    "axiom.model.dtype",
                    f"Parameter {name!r} changed dtype from {expected.get('dtype')} to {actual.get('dtype')}.",
                    field=name,
                ))

    errors = sum(1 for item in findings if item["severity"] == "error")
    warnings = sum(1 for item in findings if item["severity"] == "warning")
    return ValidationResult({
        "valid": errors == 0,
        "status": "fail" if errors else "warn" if warnings else "pass",
        "summary": {
            "errors": errors,
            "warnings": warnings,
            "columns_checked": 0,
            "checks": len(findings),
        },
        "findings": findings,
    })
