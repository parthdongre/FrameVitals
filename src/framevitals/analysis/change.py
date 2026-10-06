"""Structured Tide comparisons for graphs, tensors, and neural models."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from framevitals.analysis.tensor import tensor_metrics
from framevitals.quality_results import DriftResult


def _severity(score: float) -> str:
    if score >= 0.75:
        return "severe"
    if score >= 0.40:
        return "moderate"
    if score >= 0.15:
        return "minor"
    return "stable"


def _status(severity: str) -> str:
    if severity == "severe":
        return "fail"
    if severity in {"moderate", "minor"}:
        return "warn"
    return "pass"


def _sample_numpy(value: Any, limit: int = 100_000) -> np.ndarray:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
    else:
        detached = value.detach().reshape(-1)
        count = int(detached.numel())
        if count > limit:
            step = max(1, count // limit)
            detached = detached[::step][:limit]
        try:
            detached = detached.float()
        except Exception:
            pass
        return np.asarray(detached.cpu().numpy(), dtype=float)

    if flat.size > limit:
        step = max(1, flat.size // limit)
        flat = flat[::step][:limit]
    return np.asarray(flat, dtype=float)


def _aligned_tensor_change(reference: Any, current: Any) -> dict[str, Any]:
    ref = _sample_numpy(reference)
    cur = _sample_numpy(current)
    count = min(ref.size, cur.size)
    if count == 0:
        return {
            "sampled_values": 0,
            "relative_l2": None,
            "cosine_similarity": None,
            "wasserstein": None,
        }

    ref = ref[:count]
    cur = cur[:count]
    mask = np.isfinite(ref) & np.isfinite(cur)
    ref = ref[mask]
    cur = cur[mask]
    if not ref.size:
        return {
            "sampled_values": 0,
            "relative_l2": None,
            "cosine_similarity": None,
            "wasserstein": None,
        }

    delta = cur - ref
    ref_norm = float(np.linalg.norm(ref))
    cur_norm = float(np.linalg.norm(cur))
    denominator = max(ref_norm, 1.0e-12)
    relative_l2 = float(np.linalg.norm(delta) / denominator)
    cosine = None
    if ref_norm > 0 and cur_norm > 0:
        cosine = float(np.dot(ref, cur) / (ref_norm * cur_norm))
        cosine = max(-1.0, min(1.0, cosine))

    return {
        "sampled_values": int(ref.size),
        "relative_l2": round(relative_l2, 8),
        "cosine_similarity": round(cosine, 8) if cosine is not None else None,
        "wasserstein": round(float(wasserstein_distance(ref, cur)), 8),
        "reference_mean": round(float(np.mean(ref)), 8),
        "current_mean": round(float(np.mean(cur)), 8),
        "reference_std": round(float(np.std(ref)), 8),
        "current_std": round(float(np.std(cur)), 8),
        "mean_delta": round(float(np.mean(cur) - np.mean(ref)), 8),
        "std_delta": round(float(np.std(cur) - np.std(ref)), 8),
        "reference_zero_fraction": round(float(np.mean(np.abs(ref) <= 1.0e-8)), 8),
        "current_zero_fraction": round(float(np.mean(np.abs(cur) <= 1.0e-8)), 8),
    }


def compare_tensors(reference: Any, current: Any) -> DriftResult:
    ref_shape = tuple(int(v) for v in getattr(reference, "shape", ()))
    cur_shape = tuple(int(v) for v in getattr(current, "shape", ()))
    ref_dtype = str(getattr(reference, "dtype", "unknown"))
    cur_dtype = str(getattr(current, "dtype", "unknown"))

    aligned = _aligned_tensor_change(reference, current) if ref_shape == cur_shape else {}
    ref_metrics = tensor_metrics(reference, sample_values=100_000)
    cur_metrics = tensor_metrics(current, sample_values=100_000)

    relative_l2 = aligned.get("relative_l2")
    cosine = aligned.get("cosine_similarity")
    wasserstein = aligned.get("wasserstein")

    score = 0.0
    if ref_shape != cur_shape:
        score = max(score, 1.0)
    if ref_dtype != cur_dtype:
        score = max(score, 0.45)
    if isinstance(relative_l2, (float, int)):
        score = max(score, min(1.0, float(relative_l2)))
    if isinstance(cosine, (float, int)):
        score = max(score, min(1.0, max(0.0, 1.0 - float(cosine)) * 2.0))

    ref_matrix = ref_metrics.get("matrix") or {}
    cur_matrix = cur_metrics.get("matrix") or {}
    rank_delta = None
    if isinstance(ref_matrix, dict) and isinstance(cur_matrix, dict):
        rr = ref_matrix.get("rank_ratio")
        cr = cur_matrix.get("rank_ratio")
        if isinstance(rr, (float, int)) and isinstance(cr, (float, int)):
            rank_delta = float(cr) - float(rr)
            score = max(score, min(1.0, abs(rank_delta) * 2.0))

    severity = _severity(score)
    return DriftResult({
        "available": True,
        "source_kind": "tensor",
        "gate": {"status": _status(severity), "severity": severity},
        "summary": {
            "overall_verdict": severity,
            "shape_changed": ref_shape != cur_shape,
            "dtype_changed": ref_dtype != cur_dtype,
            "change_score": round(score, 6),
        },
        "tensor": {
            "reference_shape": list(ref_shape),
            "current_shape": list(cur_shape),
            "reference_dtype": ref_dtype,
            "current_dtype": cur_dtype,
            "aligned": aligned,
            "wasserstein": wasserstein,
            "reference_matrix": ref_matrix,
            "current_matrix": cur_matrix,
            "rank_ratio_delta": round(rank_delta, 8) if rank_delta is not None else None,
        },
        "columns": [],
    })


def _degree_distribution(graph: Any, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    degrees = np.asarray([float(degree) for _, degree in graph.degree()], dtype=float)
    if degrees.size == 0:
        return np.zeros(bins, dtype=float), np.arange(bins + 1, dtype=float)
    max_degree = max(1.0, float(np.max(degrees)))
    edges = np.linspace(0.0, max_degree + 1.0, num=bins + 1)
    hist, edges = np.histogram(degrees, bins=edges)
    probs = hist.astype(float)
    total = float(np.sum(probs))
    if total > 0:
        probs /= total
    return probs, edges


def _rebin_degrees(reference: Any, current: Any, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray([float(d) for _, d in reference.degree()], dtype=float)
    cur = np.asarray([float(d) for _, d in current.degree()], dtype=float)
    max_degree = max(
        1.0,
        float(np.max(ref)) if ref.size else 0.0,
        float(np.max(cur)) if cur.size else 0.0,
    )
    edges = np.linspace(0.0, max_degree + 1.0, num=bins + 1)
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_p = ref_hist.astype(float)
    cur_p = cur_hist.astype(float)
    if ref_p.sum():
        ref_p /= ref_p.sum()
    if cur_p.sum():
        cur_p /= cur_p.sum()
    return ref_p, cur_p


def compare_graphs(reference: Any, current: Any) -> DriftResult:
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Graph Tide requires NetworkX. Install with pip install framevitals[graph]."
        ) from exc

    from framevitals.analysis.graph import _coerce_graph

    reference, _ = _coerce_graph(reference, nx)
    current, _ = _coerce_graph(current, nx)

    ref_nodes = set(reference.nodes())
    cur_nodes = set(current.nodes())
    node_union = ref_nodes | cur_nodes
    node_jaccard = (
        len(ref_nodes & cur_nodes) / len(node_union)
        if node_union
        else 1.0
    )

    ref_n = int(reference.number_of_nodes())
    cur_n = int(current.number_of_nodes())
    ref_m = int(reference.number_of_edges())
    cur_m = int(current.number_of_edges())

    try:
        import networkx as nx

        ref_density = float(nx.density(reference))
        cur_density = float(nx.density(current))
    except Exception:
        ref_density = 0.0
        cur_density = 0.0

    ref_p, cur_p = _rebin_degrees(reference, current)
    degree_js = (
        float(jensenshannon(ref_p + 1.0e-12, cur_p + 1.0e-12, base=2.0))
        if ref_p.size
        else 0.0
    )

    directed_changed = bool(reference.is_directed()) != bool(current.is_directed())
    multigraph_changed = bool(reference.is_multigraph()) != bool(current.is_multigraph())

    node_churn = 1.0 - node_jaccard
    node_count_change = abs(cur_n - ref_n) / max(1, ref_n)
    edge_count_change = abs(cur_m - ref_m) / max(1, ref_m)
    density_change = abs(cur_density - ref_density) / max(abs(ref_density), 1.0e-12)

    score = max(
        min(1.0, node_churn * 2.0),
        min(1.0, node_count_change),
        min(1.0, edge_count_change),
        min(1.0, density_change),
        min(1.0, degree_js),
        1.0 if directed_changed else 0.0,
        0.6 if multigraph_changed else 0.0,
    )
    severity = _severity(score)

    return DriftResult({
        "available": True,
        "source_kind": "graph",
        "gate": {"status": _status(severity), "severity": severity},
        "summary": {
            "overall_verdict": severity,
            "change_score": round(score, 6),
            "nodes_added": len(cur_nodes - ref_nodes),
            "nodes_removed": len(ref_nodes - cur_nodes),
        },
        "graph": {
            "reference_nodes": ref_n,
            "current_nodes": cur_n,
            "reference_edges": ref_m,
            "current_edges": cur_m,
            "node_jaccard": round(node_jaccard, 8),
            "node_churn": round(node_churn, 8),
            "node_count_change_ratio": round(node_count_change, 8),
            "edge_count_change_ratio": round(edge_count_change, 8),
            "reference_density": round(ref_density, 10),
            "current_density": round(cur_density, 10),
            "density_change_ratio": round(density_change, 8),
            "degree_js_distance": round(degree_js, 8),
            "directed_changed": directed_changed,
            "multigraph_changed": multigraph_changed,
        },
        "columns": [],
    })


def _model_parameters(model: Any) -> dict[str, Any]:
    return {name: parameter for name, parameter in model.named_parameters()}


def compare_models(reference: Any, current: Any) -> DriftResult:
    ref_params = _model_parameters(reference)
    cur_params = _model_parameters(current)

    ref_names = set(ref_params)
    cur_names = set(cur_params)
    common = sorted(ref_names & cur_names)
    added = sorted(cur_names - ref_names)
    removed = sorted(ref_names - cur_names)

    changes: list[dict[str, Any]] = []
    shape_changes: list[dict[str, Any]] = []
    dtype_changes: list[dict[str, Any]] = []

    for name in common:
        ref = ref_params[name]
        cur = cur_params[name]
        ref_shape = tuple(int(v) for v in ref.shape)
        cur_shape = tuple(int(v) for v in cur.shape)
        if ref_shape != cur_shape:
            shape_changes.append({
                "name": name,
                "reference_shape": list(ref_shape),
                "current_shape": list(cur_shape),
            })
            continue

        ref_dtype = str(ref.dtype)
        cur_dtype = str(cur.dtype)
        if ref_dtype != cur_dtype:
            dtype_changes.append({
                "name": name,
                "reference_dtype": ref_dtype,
                "current_dtype": cur_dtype,
            })

        metrics = _aligned_tensor_change(ref, cur)
        changes.append({"name": name, **metrics})

    changes.sort(
        key=lambda item: float(item.get("relative_l2") or 0.0),
        reverse=True,
    )
    finite_changes = [
        float(item["relative_l2"])
        for item in changes
        if isinstance(item.get("relative_l2"), (int, float))
    ]
    median_change = float(np.median(finite_changes)) if finite_changes else 0.0
    p95_change = float(np.quantile(finite_changes, 0.95)) if finite_changes else 0.0
    max_change = max(finite_changes, default=0.0)

    structure_churn = (
        len(added) + len(removed) + len(shape_changes)
    ) / max(1, len(ref_names | cur_names))
    dtype_churn = len(dtype_changes) / max(1, len(common))

    score = max(
        min(1.0, p95_change),
        min(1.0, structure_churn * 2.0),
        min(1.0, dtype_churn),
    )
    severity = _severity(score)

    return DriftResult({
        "available": True,
        "source_kind": "model",
        "gate": {"status": _status(severity), "severity": severity},
        "summary": {
            "overall_verdict": severity,
            "change_score": round(score, 6),
            "parameters_compared": len(changes),
            "parameters_added": len(added),
            "parameters_removed": len(removed),
            "shape_changes": len(shape_changes),
            "dtype_changes": len(dtype_changes),
        },
        "model": {
            "reference_class": type(reference).__name__,
            "current_class": type(current).__name__,
            "parameters_compared": len(changes),
            "added_parameters": added,
            "removed_parameters": removed,
            "shape_changes": shape_changes,
            "dtype_changes": dtype_changes,
            "median_relative_l2": round(median_change, 8),
            "p95_relative_l2": round(p95_change, 8),
            "max_relative_l2": round(max_change, 8),
            "most_changed": changes[:25],
        },
        "columns": [],
    })
