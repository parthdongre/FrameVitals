"""ONNX model diagnostics built on optional onnx dependency."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


from framevitals.core.beacons import beacon
from framevitals.quality_results import DriftResult
from framevitals.result import AnalysisResult


def _require_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "ONNX diagnostics require the optional onnx package. "
            "Install with pip install framevitals[onnx]."
        ) from exc
    return onnx


def _dtype_name(onnx: Any, data_type: int) -> str:
    try:
        return str(onnx.TensorProto.DataType.Name(data_type))
    except Exception:
        return str(data_type)


def _shape_from_value_info(value: Any) -> list[Any]:
    try:
        tensor_type = value.type.tensor_type
        dims = tensor_type.shape.dim
    except Exception:
        return []
    shape: list[Any] = []
    for dim in dims:
        if getattr(dim, "dim_value", 0):
            shape.append(int(dim.dim_value))
        elif getattr(dim, "dim_param", ""):
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return shape


def _initializer_summary(onnx: Any, initializer: Any) -> dict[str, Any]:
    shape = [int(v) for v in initializer.dims]
    elements = int(math.prod(shape)) if shape else 1
    dtype = _dtype_name(onnx, int(initializer.data_type))
    return {
        "name": str(initializer.name),
        "shape": shape,
        "elements": elements,
        "dtype": dtype,
        "raw_bytes": len(getattr(initializer, "raw_data", b"") or b""),
    }


def _architecture(operator_counts: Counter[str], names: list[str]) -> str:
    ops = set(operator_counts)
    lowered = [name.lower() for name in names]
    if {"Conv", "MaxPool"} & ops or "Conv" in ops:
        return "cnn"
    if "Attention" in ops or any(
        marker in name
        for name in lowered
        for marker in ("q_proj", "k_proj", "v_proj", "attention", "self_attn")
    ):
        return "transformer"
    if {"LSTM", "GRU", "RNN"} & ops:
        return "recurrent"
    return "computation_graph"


def inspect_onnx(path: str | Path) -> dict[str, Any]:
    onnx = _require_onnx()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    model = onnx.load(str(source), load_external_data=False)
    graph = model.graph

    nodes = list(graph.node)
    initializers = [_initializer_summary(onnx, item) for item in graph.initializer]
    initializer_names = {item["name"] for item in initializers}
    operator_counts = Counter(str(node.op_type) for node in nodes)

    producers: dict[str, int] = {}
    consumers: defaultdict[str, list[int]] = defaultdict(list)
    node_names: list[str] = []
    duplicate_node_names: list[str] = []
    seen_names: set[str] = set()

    for index, node in enumerate(nodes):
        name = str(node.name or f"{node.op_type}:{index}")
        node_names.append(name)
        if node.name:
            if node.name in seen_names:
                duplicate_node_names.append(str(node.name))
            seen_names.add(str(node.name))

        for output in node.output:
            if output:
                producers[str(output)] = index
        for inp in node.input:
            if inp:
                consumers[str(inp)].append(index)

    graph_inputs = [str(item.name) for item in graph.input]
    graph_outputs = [str(item.name) for item in graph.output]

    edges: set[tuple[int, int]] = set()
    indegree = [0] * len(nodes)
    outdegree = [0] * len(nodes)
    for tensor_name, producer_idx in producers.items():
        for consumer_idx in consumers.get(tensor_name, []):
            if producer_idx == consumer_idx:
                continue
            edge = (producer_idx, consumer_idx)
            if edge not in edges:
                edges.add(edge)
                outdegree[producer_idx] += 1
                indegree[consumer_idx] += 1

    queue = deque(i for i, degree in enumerate(indegree) if degree == 0)
    visited = 0
    mutable_indegree = list(indegree)
    adjacency: defaultdict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)

    while queue:
        current = queue.popleft()
        visited += 1
        for nxt in adjacency.get(current, []):
            mutable_indegree[nxt] -= 1
            if mutable_indegree[nxt] == 0:
                queue.append(nxt)

    acyclic = visited == len(nodes)

    output_names = set(graph_outputs)
    dead_end_nodes = [
        node_names[index]
        for index, degree in enumerate(outdegree)
        if degree == 0
        and not any(str(output) in output_names for output in nodes[index].output)
    ]

    referenced_inputs = {
        str(inp)
        for node in nodes
        for inp in node.input
        if inp
    }
    unused_initializers = sorted(initializer_names - referenced_inputs)

    declared_input_names = set(graph_inputs)
    truly_external_inputs = sorted(declared_input_names - initializer_names)

    value_infos = {
        str(item.name): _shape_from_value_info(item)
        for item in list(graph.input) + list(graph.output) + list(graph.value_info)
    }

    total_parameters = sum(int(item["elements"]) for item in initializers)
    dtype_counts: Counter[str] = Counter()
    for item in initializers:
        dtype_counts[item["dtype"]] += int(item["elements"])

    disconnected_nodes = 0
    if nodes:
        undirected: defaultdict[int, set[int]] = defaultdict(set)
        for left, right in edges:
            undirected[left].add(right)
            undirected[right].add(left)
        seen: set[int] = set()
        components = 0
        largest_component = 0
        for start in range(len(nodes)):
            if start in seen:
                continue
            components += 1
            stack = [start]
            size = 0
            seen.add(start)
            while stack:
                current = stack.pop()
                size += 1
                for nxt in undirected.get(current, set()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            largest_component = max(largest_component, size)
        disconnected_nodes = len(nodes) - largest_component
    else:
        components = 0
        largest_component = 0

    architecture = _architecture(operator_counts, [item["name"] for item in initializers])

    return {
        "format": "onnx",
        "path": str(source),
        "filename": source.name,
        "file_size": source.stat().st_size,
        "ir_version": int(getattr(model, "ir_version", 0)),
        "producer_name": str(getattr(model, "producer_name", "") or ""),
        "producer_version": str(getattr(model, "producer_version", "") or ""),
        "opset_imports": {
            str(item.domain or "ai.onnx"): int(item.version)
            for item in model.opset_import
        },
        "architecture": architecture,
        "nodes": len(nodes),
        "edges": len(edges),
        "acyclic": bool(acyclic),
        "components": components,
        "largest_component_nodes": largest_component,
        "disconnected_nodes": disconnected_nodes,
        "graph_inputs": graph_inputs,
        "external_inputs": truly_external_inputs,
        "graph_outputs": graph_outputs,
        "operator_counts": dict(operator_counts.most_common()),
        "duplicate_node_names": sorted(set(duplicate_node_names)),
        "dead_end_nodes": dead_end_nodes,
        "initializers": initializers,
        "initializer_count": len(initializers),
        "unused_initializers": unused_initializers,
        "parameters": total_parameters,
        "dtype_parameter_counts": dict(dtype_counts),
        "value_shapes": value_infos,
    }


def _health_label(score: float) -> str:
    if score >= 90:
        return "healthy"
    if score >= 75:
        return "good"
    if score >= 55:
        return "attention"
    return "critical"


def analyze_onnx(path: str | Path, *, depth: str | None = None) -> AnalysisResult:
    summary = inspect_onnx(path)
    findings: list[dict[str, Any]] = []

    if not summary["acyclic"]:
        findings.append(beacon(
            "model.onnx.cycle",
            "The ONNX operator graph contains a cycle",
            severity="critical",
            confidence=1.0,
            recommendation="Validate the exported graph and control-flow operators.",
        ))

    if summary["unused_initializers"]:
        findings.append(beacon(
            "model.onnx.unused_initializers",
            "Unused parameter initializers were found",
            severity="medium",
            confidence=1.0,
            summary=f"{len(summary['unused_initializers'])} initializer tensors are never consumed.",
            recommendation="Remove stale parameters or verify the export path.",
            evidence={"initializers": summary["unused_initializers"][:50]},
        ))

    if summary["dead_end_nodes"]:
        findings.append(beacon(
            "model.onnx.dead_ends",
            "Dead-end operators were found",
            severity="medium",
            confidence=0.98,
            summary=f"{len(summary['dead_end_nodes'])} nodes do not feed a declared graph output.",
            recommendation="Inspect pruning/export logic and disconnected computation branches.",
            evidence={"nodes": summary["dead_end_nodes"][:50]},
        ))

    if summary["disconnected_nodes"]:
        findings.append(beacon(
            "model.onnx.disconnected",
            "The operator graph contains disconnected regions",
            severity="high",
            confidence=1.0,
            summary=f"{summary['disconnected_nodes']} nodes sit outside the largest graph component.",
            recommendation="Inspect disconnected branches and export-time artifacts.",
            evidence={
                "components": summary["components"],
                "disconnected_nodes": summary["disconnected_nodes"],
            },
        ))

    if summary["duplicate_node_names"]:
        findings.append(beacon(
            "model.onnx.duplicate_node_names",
            "Duplicate ONNX node names were detected",
            severity="medium",
            confidence=1.0,
            summary=f"{len(summary['duplicate_node_names'])} node names are duplicated.",
            recommendation="Use unique node names to improve debugging and graph traceability.",
            evidence={"names": summary["duplicate_node_names"][:50]},
        ))

    score = 100.0
    if not summary["acyclic"]:
        score -= 50.0
    score -= min(20.0, len(summary["unused_initializers"]) * 2.0)
    score -= min(25.0, len(summary["dead_end_nodes"]) * 3.0)
    score -= min(30.0, int(summary["disconnected_nodes"]) * 2.0)
    score -= min(15.0, len(summary["duplicate_node_names"]) * 2.0)
    score = round(max(0.0, min(100.0, score)), 2)

    model_summary = {
        **summary,
        "framework": "onnx",
        "class_name": "ONNXModel",
        "trainable_parameters": None,
        "frozen_parameters": None,
        "runtime": None,
    }

    return AnalysisResult({
        "dataset_id": None,
        "filename": summary["filename"],
        "analysis_mode": str(depth or "standard"),
        "source_kind": "model",
        "profile": {
            "shape": {
                "rows": int(summary["parameters"]),
                "columns": int(summary["nodes"]),
            },
            "structure": model_summary,
        },
        "model": model_summary,
        "health": {"overall_score": score, "label": _health_label(score)},
        "ml_readiness": {"score": None, "label": "not_applicable"},
        "findings": findings,
        "artifacts_enabled": False,
        "execution": {
            "method": "onnx_graph_diagnostics",
            "sampled": False,
            "resource_bounded": True,
            "tensor_payloads_loaded": False,
        },
    })


def compare_onnx(reference: str | Path, current: str | Path) -> DriftResult:
    ref = inspect_onnx(reference)
    cur = inspect_onnx(current)

    ref_ops = Counter(ref["operator_counts"])
    cur_ops = Counter(cur["operator_counts"])
    all_ops = sorted(set(ref_ops) | set(cur_ops))
    op_changes = {
        op: {"reference": int(ref_ops.get(op, 0)), "current": int(cur_ops.get(op, 0))}
        for op in all_ops
        if ref_ops.get(op, 0) != cur_ops.get(op, 0)
    }

    ref_init = {item["name"]: item for item in ref["initializers"]}
    cur_init = {item["name"]: item for item in cur["initializers"]}
    ref_names = set(ref_init)
    cur_names = set(cur_init)

    added = sorted(cur_names - ref_names)
    removed = sorted(ref_names - cur_names)
    shape_changes: list[dict[str, Any]] = []
    dtype_changes: list[dict[str, Any]] = []
    for name in sorted(ref_names & cur_names):
        left = ref_init[name]
        right = cur_init[name]
        if left["shape"] != right["shape"]:
            shape_changes.append({
                "name": name,
                "reference_shape": left["shape"],
                "current_shape": right["shape"],
            })
        if left["dtype"] != right["dtype"]:
            dtype_changes.append({
                "name": name,
                "reference_dtype": left["dtype"],
                "current_dtype": right["dtype"],
            })

    node_change = abs(cur["nodes"] - ref["nodes"]) / max(1, ref["nodes"])
    edge_change = abs(cur["edges"] - ref["edges"]) / max(1, ref["edges"])
    param_change = abs(cur["parameters"] - ref["parameters"]) / max(1, ref["parameters"])
    structure_churn = (
        len(added) + len(removed) + len(shape_changes) + len(op_changes)
    ) / max(1, len(ref_names | cur_names) + len(all_ops))

    score = max(
        min(1.0, node_change),
        min(1.0, edge_change),
        min(1.0, param_change),
        min(1.0, structure_churn * 2.0),
    )
    severity = "severe" if score >= 0.75 else "moderate" if score >= 0.40 else "minor" if score >= 0.15 else "stable"
    status = "fail" if severity == "severe" else "warn" if severity != "stable" else "pass"

    return DriftResult({
        "available": True,
        "source_kind": "model",
        "gate": {"status": status, "severity": severity},
        "summary": {
            "overall_verdict": severity,
            "change_score": round(score, 6),
            "parameters_compared": len(ref_names & cur_names),
            "parameters_added": len(added),
            "parameters_removed": len(removed),
            "shape_changes": len(shape_changes),
            "dtype_changes": len(dtype_changes),
        },
        "model": {
            "reference_class": "ONNXModel",
            "current_class": "ONNXModel",
            "reference_parameters": ref["parameters"],
            "current_parameters": cur["parameters"],
            "parameters_compared": len(ref_names & cur_names),
            "added_parameters": added,
            "removed_parameters": removed,
            "shape_changes": shape_changes,
            "dtype_changes": dtype_changes,
            "operator_changes": op_changes,
            "reference_nodes": ref["nodes"],
            "current_nodes": cur["nodes"],
            "reference_edges": ref["edges"],
            "current_edges": cur["edges"],
            "comparison_scope": "graph_and_metadata",
            "median_relative_l2": None,
            "p95_relative_l2": None,
            "max_relative_l2": None,
            "most_changed": [],
        },
        "columns": [],
    })
