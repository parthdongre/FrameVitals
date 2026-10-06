"""Bounded graph diagnostics for NetworkX-compatible graph objects.

The engine deliberately uses an algorithm portfolio instead of blindly running
all graph algorithms. Expensive work is bounded by graph size and requested
Prism depth so diagnostics remain useful on real networks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framevitals.core.beacons import beacon
from framevitals.result import AnalysisResult


_DEPTH_BUDGETS = {
    "quick": {
        "path_sources": 6,
        "betweenness_sources": 12,
        "centrality_edges": 80_000,
        "community_edges": 40_000,
        "cut_edges": 60_000,
        "clustering_edges": 40_000,
    },
    "standard": {
        "path_sources": 12,
        "betweenness_sources": 24,
        "centrality_edges": 250_000,
        "community_edges": 150_000,
        "cut_edges": 200_000,
        "clustering_edges": 120_000,
    },
    "deep": {
        "path_sources": 24,
        "betweenness_sources": 48,
        "centrality_edges": 600_000,
        "community_edges": 400_000,
        "cut_edges": 500_000,
        "clustering_edges": 300_000,
    },
    "research": {
        "path_sources": 48,
        "betweenness_sources": 96,
        "centrality_edges": 1_500_000,
        "community_edges": 1_000_000,
        "cut_edges": 1_000_000,
        "clustering_edges": 750_000,
    },
}


def _budget(depth: str | None) -> dict[str, int]:
    return _DEPTH_BUDGETS.get(str(depth or "standard").lower(), _DEPTH_BUDGETS["standard"])


def _even_sample(values: list[Any], count: int) -> list[Any]:
    if len(values) <= count:
        return list(values)
    positions = np.linspace(0, len(values) - 1, num=max(1, count), dtype=int)
    return [values[int(i)] for i in positions]


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _detect_weight_attribute(graph: Any, *, sample_edges: int = 256) -> str | None:
    """Use a conventional non-negative numeric weight attribute when it is reliable."""
    checked = 0
    valid = 0
    negative = 0
    try:
        iterator = graph.edges(data=True)
    except Exception:
        return None

    for _, _, data in iterator:
        checked += 1
        value = data.get("weight") if isinstance(data, dict) else None
        numeric = _safe_float(value)
        if numeric is not None:
            if numeric < 0:
                negative += 1
            else:
                valid += 1
        if checked >= sample_edges:
            break

    if checked == 0 or negative:
        return None
    return "weight" if valid / checked >= 0.80 else None


def _health_label(score: float) -> str:
    if score >= 90:
        return "healthy"
    if score >= 75:
        return "good"
    if score >= 55:
        return "attention"
    return "critical"


def _node_label(value: Any) -> str:
    rendered = str(value)
    return rendered if len(rendered) <= 160 else rendered[:157] + "..."


def _top_items(scores: dict[Any, float], limit: int = 10) -> list[dict[str, Any]]:
    ordered = sorted(scores.items(), key=lambda item: (-float(item[1]), repr(item[0])))
    return [
        {"node": _node_label(node), "score": round(float(score), 8)}
        for node, score in ordered[:limit]
    ]


def _component_summary(graph: Any, nx: Any) -> dict[str, Any]:
    n = graph.number_of_nodes()
    if n == 0:
        return {
            "component_count": 0,
            "largest_component_nodes": 0,
            "largest_component_ratio": 0.0,
        }

    if graph.is_directed():
        weak_sizes = sorted((len(c) for c in nx.weakly_connected_components(graph)), reverse=True)
        strong_sizes = sorted((len(c) for c in nx.strongly_connected_components(graph)), reverse=True)
        return {
            "component_kind": "weak",
            "component_count": len(weak_sizes),
            "largest_component_nodes": weak_sizes[0] if weak_sizes else 0,
            "largest_component_ratio": round((weak_sizes[0] if weak_sizes else 0) / n, 6),
            "strong_component_count": len(strong_sizes),
            "largest_strong_component_nodes": strong_sizes[0] if strong_sizes else 0,
            "largest_strong_component_ratio": round((strong_sizes[0] if strong_sizes else 0) / n, 6),
        }

    sizes = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
    return {
        "component_kind": "connected",
        "component_count": len(sizes),
        "largest_component_nodes": sizes[0] if sizes else 0,
        "largest_component_ratio": round((sizes[0] if sizes else 0) / n, 6),
    }


def _degree_summary(graph: Any) -> tuple[dict[str, Any], list[Any]]:
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    if degrees.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "std": 0.0,
        }, []

    median = float(np.median(degrees))
    mad = float(np.median(np.abs(degrees - median)))
    threshold = median + 8.0 * max(mad, 1.0)
    hubs = [
        node
        for node, degree in graph.degree()
        if float(degree) > threshold and float(degree) >= max(10.0, median * 4.0)
    ]
    return {
        "mean": round(float(np.mean(degrees)), 6),
        "median": round(median, 6),
        "p95": round(float(np.quantile(degrees, 0.95)), 6),
        "max": round(float(np.max(degrees)), 6),
        "std": round(float(np.std(degrees)), 6),
        "robust_hub_threshold": round(float(threshold), 6),
    }, hubs


def _sample_shortest_paths(
    graph: Any,
    nx: Any,
    *,
    source_count: int,
    weight: str | None,
) -> dict[str, Any]:
    nodes = list(graph.nodes())
    if not nodes:
        return {"available": False, "reason": "empty_graph"}

    sources = _even_sample(nodes, min(source_count, len(nodes)))
    distances: list[float] = []
    reached = 0
    possible = 0

    for source in sources:
        try:
            if weight:
                lengths = nx.single_source_dijkstra_path_length(
                    graph,
                    source,
                    weight=weight,
                )
            else:
                lengths = nx.single_source_shortest_path_length(graph, source)
        except Exception as exc:
            return {
                "available": False,
                "reason": type(exc).__name__,
                "weighted": bool(weight),
                "weight_attribute": weight,
            }
        for node, distance in lengths.items():
            if node == source:
                continue
            d = _safe_float(distance)
            if d is not None:
                distances.append(d)
                reached += 1
        possible += max(0, len(nodes) - 1)

    if not distances:
        return {
            "available": True,
            "sources": len(sources),
            "reachable_ratio": 0.0,
            "mean_distance": None,
            "p95_distance": None,
            "max_distance": None,
            "weighted": bool(weight),
        }

    arr = np.asarray(distances, dtype=float)
    return {
        "available": True,
        "sources": len(sources),
        "reachable_ratio": round(reached / possible, 6) if possible else 1.0,
        "mean_distance": round(float(np.mean(arr)), 6),
        "p95_distance": round(float(np.quantile(arr, 0.95)), 6),
        "max_distance": round(float(np.max(arr)), 6),
        "weighted": bool(weight),
        "weight_attribute": weight,
    }


def _centrality_summary(
    graph: Any,
    nx: Any,
    budget: dict[str, int],
    *,
    weight: str | None = None,
) -> dict[str, Any]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    result: dict[str, Any] = {}

    if not n:
        return result

    if m <= budget["centrality_edges"]:
        try:
            pagerank = nx.pagerank(
                graph,
                max_iter=100,
                tol=1.0e-6,
                weight=weight,
            )
            result["pagerank"] = {"top": _top_items(pagerank)}
        except Exception as exc:
            result["pagerank"] = {"available": False, "reason": type(exc).__name__}

        k = min(budget["betweenness_sources"], n)
        try:
            betweenness = nx.betweenness_centrality(
                graph,
                k=k if k < n else None,
                normalized=True,
                weight=weight,
                seed=42,
            )
            result["betweenness"] = {
                "sampled_sources": k if k < n else n,
                "approximate": bool(k < n),
                "top": _top_items(betweenness),
            }
        except Exception as exc:
            result["betweenness"] = {"available": False, "reason": type(exc).__name__}
    else:
        result["pagerank"] = {
            "available": False,
            "reason": "resource_budget",
            "edge_limit": budget["centrality_edges"],
        }
        result["betweenness"] = {
            "available": False,
            "reason": "resource_budget",
            "edge_limit": budget["centrality_edges"],
        }

    return result


def _cut_structure(graph: Any, nx: Any, budget: dict[str, int]) -> dict[str, Any]:
    m = graph.number_of_edges()
    if m > budget["cut_edges"]:
        return {
            "available": False,
            "reason": "resource_budget",
            "edge_limit": budget["cut_edges"],
        }

    try:
        simple = nx.Graph(graph)
        if simple.number_of_nodes() == 0:
            return {"available": True, "bridges": 0, "articulation_points": 0}
        bridges = list(nx.bridges(simple))
        articulation = list(nx.articulation_points(simple))
        return {
            "available": True,
            "bridges": len(bridges),
            "articulation_points": len(articulation),
            "top_articulation_points": [_node_label(v) for v in articulation[:20]],
        }
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


def _community_summary(
    graph: Any,
    nx: Any,
    budget: dict[str, int],
    *,
    weight: str | None = None,
) -> dict[str, Any]:
    m = graph.number_of_edges()
    if m > budget["community_edges"] or graph.number_of_nodes() < 2:
        return {
            "available": False,
            "reason": "resource_budget" if m > budget["community_edges"] else "too_small",
            "edge_limit": budget["community_edges"],
        }

    simple = nx.Graph(graph)
    try:
        louvain = getattr(nx.algorithms.community, "louvain_communities", None)
        if callable(louvain):
            communities = list(louvain(simple, seed=42, weight=weight))
            method = "louvain"
        else:
            communities = list(
                nx.algorithms.community.greedy_modularity_communities(
                    simple,
                    weight=weight,
                )
            )
            method = "greedy_modularity"
        sizes = sorted((len(c) for c in communities), reverse=True)
        modularity = (
            nx.algorithms.community.modularity(
                simple,
                communities,
                weight=weight,
            )
            if communities and simple.number_of_edges()
            else 0.0
        )
        return {
            "available": True,
            "method": method,
            "count": len(communities),
            "largest_sizes": sizes[:20],
            "modularity": round(float(modularity), 6),
        }
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


def _clustering_summary(graph: Any, nx: Any, budget: dict[str, int]) -> dict[str, Any]:
    m = graph.number_of_edges()
    if m > budget["clustering_edges"]:
        return {
            "available": False,
            "reason": "resource_budget",
            "edge_limit": budget["clustering_edges"],
        }
    try:
        simple = nx.Graph(graph)
        return {
            "available": True,
            "average_clustering": round(float(nx.average_clustering(simple)), 6),
            "transitivity": round(float(nx.transitivity(simple)), 6),
        }
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


def analyze_graph(
    graph: Any,
    *,
    depth: str | None = None,
    weight: str | None = None,
) -> AnalysisResult:
    """Run structural diagnostics on a NetworkX graph."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Graph diagnostics require NetworkX. Install with "
            "pip install framevitals[graph]."
        ) from exc

    if not (
        callable(getattr(graph, "number_of_nodes", None))
        and callable(getattr(graph, "number_of_edges", None))
    ):
        raise TypeError("Expected a NetworkX-compatible graph object.")

    budget = _budget(depth)
    n = int(graph.number_of_nodes())
    m = int(graph.number_of_edges())
    directed = bool(graph.is_directed())
    multigraph = bool(graph.is_multigraph())

    components = _component_summary(graph, nx)
    degrees, hubs = _degree_summary(graph)
    isolates = list(nx.isolates(graph)) if n else []
    self_loops = int(nx.number_of_selfloops(graph)) if n else 0
    density = float(nx.density(graph)) if n > 1 else 0.0

    resolved_weight = weight or _detect_weight_attribute(graph)
    paths = _sample_shortest_paths(
        graph,
        nx,
        source_count=budget["path_sources"],
        weight=resolved_weight,
    )
    centrality = _centrality_summary(
        graph,
        nx,
        budget,
        weight=resolved_weight,
    )
    cuts = _cut_structure(graph, nx, budget)
    communities = _community_summary(
        graph,
        nx,
        budget,
        weight=resolved_weight,
    )
    clustering = _clustering_summary(graph, nx, budget)

    findings: list[dict[str, Any]] = []
    largest_ratio = float(components.get("largest_component_ratio") or 0.0)
    isolate_ratio = len(isolates) / n if n else 0.0
    hub_ratio = len(hubs) / n if n else 0.0

    if n and largest_ratio < 0.95:
        findings.append(
            beacon(
                "graph.fragmentation",
                "The graph is structurally fragmented",
                severity="high" if largest_ratio < 0.80 else "medium",
                confidence=1.0,
                summary=(
                    f"The largest component contains {largest_ratio:.1%} of nodes "
                    f"across {components.get('component_count', 0)} components."
                ),
                recommendation="Inspect disconnected components and ingestion/join boundaries.",
                evidence={
                    "largest_component_ratio": round(largest_ratio, 6),
                    "component_count": components.get("component_count"),
                },
            )
        )

    if isolate_ratio >= 0.01:
        findings.append(
            beacon(
                "graph.isolates",
                "A meaningful share of nodes are isolated",
                severity="high" if isolate_ratio >= 0.10 else "medium",
                summary=f"{len(isolates):,} nodes ({isolate_ratio:.1%}) have degree zero.",
                recommendation="Check orphan entities, missing relationships, or filtering errors.",
                evidence={"isolates": len(isolates), "ratio": round(isolate_ratio, 6)},
            )
        )

    if self_loops:
        findings.append(
            beacon(
                "graph.self_loops",
                "Self-loops are present",
                severity="medium",
                summary=f"{self_loops:,} self-loop edges were found.",
                recommendation="Confirm whether self-relations are valid for this network.",
                evidence={"self_loops": self_loops},
            )
        )

    if hubs:
        findings.append(
            beacon(
                "graph.hubs",
                "Extreme hub nodes were detected",
                severity="high" if hub_ratio >= 0.01 else "medium",
                confidence=0.95,
                summary=(
                    f"{len(hubs):,} nodes exceed the robust degree threshold "
                    f"({degrees.get('robust_hub_threshold')})."
                ),
                recommendation="Inspect whether these hubs are legitimate, duplicated, or anomalous.",
                evidence={
                    "hub_count": len(hubs),
                    "hub_ratio": round(hub_ratio, 6),
                    "examples": [_node_label(v) for v in hubs[:20]],
                },
            )
        )

    reachable_ratio = paths.get("reachable_ratio")
    if isinstance(reachable_ratio, (float, int)) and reachable_ratio < 0.85:
        findings.append(
            beacon(
                "graph.reachability",
                "Sampled shortest-path reachability is low",
                severity="high" if reachable_ratio < 0.60 else "medium",
                confidence=0.9,
                summary=f"Only {float(reachable_ratio):.1%} of sampled node pairs were reachable.",
                recommendation="Inspect directionality, component boundaries, and missing edges.",
                evidence=dict(paths),
            )
        )

    if cuts.get("available") and cuts.get("articulation_points", 0):
        articulation_count = int(cuts["articulation_points"])
        articulation_ratio = articulation_count / n if n else 0.0
        if articulation_ratio >= 0.001 or articulation_count >= 5:
            findings.append(
                beacon(
                    "graph.structural_bottlenecks",
                    "Structural bottleneck nodes were found",
                    severity="medium",
                    confidence=0.95,
                    summary=(
                        f"{articulation_count:,} articulation points and "
                        f"{int(cuts.get('bridges', 0)):,} bridges can disconnect the network."
                    ),
                    recommendation="Review critical bridge nodes/edges for resilience or data errors.",
                    evidence=dict(cuts),
                )
            )

    score = 100.0
    score -= min(35.0, max(0.0, (0.98 - largest_ratio) * 100.0)) if n else 0.0
    score -= min(20.0, isolate_ratio * 150.0)
    score -= min(12.0, self_loops / max(m, 1) * 100.0)
    score -= min(12.0, hub_ratio * 200.0)
    if isinstance(reachable_ratio, (float, int)):
        score -= min(20.0, max(0.0, (0.95 - float(reachable_ratio)) * 50.0))
    score = round(max(0.0, min(100.0, score)), 2)

    graph_summary = {
        "nodes": n,
        "edges": m,
        "directed": directed,
        "multigraph": multigraph,
        "density": round(density, 8),
        "isolates": len(isolates),
        "self_loops": self_loops,
        "weight_attribute": resolved_weight,
        "degree": degrees,
        "components": components,
        "shortest_paths": paths,
        "centrality": centrality,
        "cut_structure": cuts,
        "communities": communities,
        "clustering": clustering,
        "algorithm_budget": dict(budget),
    }

    return AnalysisResult(
        {
            "dataset_id": None,
            "filename": type(graph).__name__,
            "analysis_mode": str(depth or "standard"),
            "source_kind": "graph",
            "profile": {
                "shape": {"rows": n, "columns": m},
                "structure": graph_summary,
            },
            "graph": graph_summary,
            "health": {
                "overall_score": score,
                "label": _health_label(score),
            },
            "ml_readiness": {"score": None, "label": "not_applicable"},
            "findings": findings,
            "artifacts_enabled": False,
            "execution": {
                "method": "bounded_graph_diagnostics",
                "sampled": True,
                "resource_bounded": True,
            },
        }
    )
