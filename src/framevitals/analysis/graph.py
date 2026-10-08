"""Bounded graph diagnostics for NetworkX-compatible graph objects.

The engine deliberately uses an algorithm portfolio instead of blindly running
all graph algorithms. Expensive work is bounded by graph size and requested
Prism depth so diagnostics remain useful on real networks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from framevitals.core.beacons import beacon
from framevitals.result import AnalysisResult


_DEPTH_BUDGETS = {
    "quick": {
        "path_sources": 6,
        "path_nodes": 100_000,
        "path_edges": 250_000,
        "betweenness_sources": 12,
        "centrality_nodes": 20_000,
        "centrality_edges": 80_000,
        "community_nodes": 20_000,
        "community_edges": 40_000,
        "cut_nodes": 50_000,
        "cut_edges": 60_000,
        "clustering_nodes": 20_000,
        "clustering_edges": 40_000,
        "spectral_nodes": 2_000,
        "spectral_edges": 20_000,
        "core_nodes": 50_000,
        "core_edges": 100_000,
    },
    "standard": {
        "path_sources": 12,
        "path_nodes": 250_000,
        "path_edges": 600_000,
        "betweenness_sources": 24,
        "centrality_nodes": 75_000,
        "centrality_edges": 250_000,
        "community_nodes": 75_000,
        "community_edges": 150_000,
        "cut_nodes": 150_000,
        "cut_edges": 200_000,
        "clustering_nodes": 60_000,
        "clustering_edges": 120_000,
        "spectral_nodes": 5_000,
        "spectral_edges": 60_000,
        "core_nodes": 150_000,
        "core_edges": 300_000,
    },
    "deep": {
        "path_sources": 24,
        "path_nodes": 500_000,
        "path_edges": 1_500_000,
        "betweenness_sources": 48,
        "centrality_nodes": 200_000,
        "centrality_edges": 600_000,
        "community_nodes": 200_000,
        "community_edges": 400_000,
        "cut_nodes": 350_000,
        "cut_edges": 500_000,
        "clustering_nodes": 150_000,
        "clustering_edges": 300_000,
        "spectral_nodes": 10_000,
        "spectral_edges": 150_000,
        "core_nodes": 300_000,
        "core_edges": 750_000,
    },
    "research": {
        "path_sources": 48,
        "path_nodes": 1_000_000,
        "path_edges": 3_000_000,
        "betweenness_sources": 96,
        "centrality_nodes": 500_000,
        "centrality_edges": 1_500_000,
        "community_nodes": 500_000,
        "community_edges": 1_000_000,
        "cut_nodes": 750_000,
        "cut_edges": 1_000_000,
        "clustering_nodes": 300_000,
        "clustering_edges": 750_000,
        "spectral_nodes": 20_000,
        "spectral_edges": 400_000,
        "core_nodes": 750_000,
        "core_edges": 2_000_000,
    },
}


def _budget(depth: str | None) -> dict[str, int]:
    return _DEPTH_BUDGETS.get(str(depth or "standard").lower(), _DEPTH_BUDGETS["standard"])


def _coerce_graph(graph: Any, nx: Any) -> tuple[Any, str | None]:
    """Load supported graph files or return an existing graph object."""
    if isinstance(graph, (str, Path)):
        path = Path(graph)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".graphml":
            return nx.read_graphml(path), path.name
        if suffix == ".gexf":
            return nx.read_gexf(path), path.name
        if suffix == ".gml":
            return nx.read_gml(path), path.name
        raise ValueError(f"Unsupported graph file format: {suffix or '<none>'}")
    return graph, None


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


def _inspect_edge_weights(graph: Any, *, attribute: str = "weight") -> dict[str, Any]:
    """Validate all edge costs in O(E) time and constant additional memory.

    A sampled check can miss a later negative edge, making Dijkstra invalid.
    Missing edge weights are interpreted as unit costs by NetworkX.
    """
    checked = valid = invalid = negative = nonnumeric = missing = 0
    try:
        iterator = graph.edges(data=True)
    except Exception:
        return {"selected": None, "status": "unavailable"}

    for _, _, data in iterator:
        checked += 1
        if not isinstance(data, dict) or attribute not in data:
            missing += 1
            continue
        numeric = _safe_float(data[attribute])
        if numeric is None:
            invalid += 1
            nonnumeric += 1
        elif numeric < 0:
            invalid += 1
            negative += 1
        else:
            valid += 1

    selected = attribute if checked and not invalid and valid / checked >= 0.80 else None
    return {
        "selected": selected,
        "status": "invalid" if invalid else "selected" if selected else "unweighted",
        "attribute": attribute,
        "edges_checked": checked,
        "edges_with_valid_weights": valid,
        "edges_without_weights": missing,
        "invalid_weights": invalid,
        "negative_weights": negative,
        "nonnumeric_or_nonfinite_weights": nonnumeric,
    }


def _detect_weight_attribute(graph: Any) -> str | None:
    """Compatibility helper for the conventional NetworkX weight attribute."""
    return _inspect_edge_weights(graph)["selected"]


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

    if n <= budget["centrality_nodes"] and m <= budget["centrality_edges"]:
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
            "node_limit": budget["centrality_nodes"],
            "edge_limit": budget["centrality_edges"],
        }
        result["betweenness"] = {
            "available": False,
            "reason": "resource_budget",
            "node_limit": budget["centrality_nodes"],
            "edge_limit": budget["centrality_edges"],
        }

    return result


def _cut_structure(graph: Any, nx: Any, budget: dict[str, int]) -> dict[str, Any]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n > budget["cut_nodes"] or m > budget["cut_edges"]:
        return {
            "available": False,
            "reason": "resource_budget",
            "node_limit": budget["cut_nodes"],
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
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    over_budget = (
        n > budget["community_nodes"]
        or m > budget["community_edges"]
    )
    if over_budget or n < 2:
        return {
            "available": False,
            "reason": "resource_budget" if over_budget else "too_small",
            "node_limit": budget["community_nodes"],
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
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n > budget["clustering_nodes"] or m > budget["clustering_edges"]:
        return {
            "available": False,
            "reason": "resource_budget",
            "node_limit": budget["clustering_nodes"],
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


def _core_summary(graph: Any, nx: Any, budget: dict[str, int]) -> dict[str, Any]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n > budget["core_nodes"] or m > budget["core_edges"]:
        return {
            "available": False,
            "reason": "resource_budget",
            "node_limit": budget["core_nodes"],
            "edge_limit": budget["core_edges"],
        }
    try:
        simple = nx.Graph(graph)
        simple.remove_edges_from(list(nx.selfloop_edges(simple)))
        if not simple.number_of_nodes():
            return {"available": True, "max_core": 0, "degeneracy": 0}
        core = nx.core_number(simple)
        max_core = max(core.values(), default=0)
        counts: dict[int, int] = {}
        for value in core.values():
            counts[int(value)] = counts.get(int(value), 0) + 1
        assortativity = None
        if simple.number_of_edges() > 0:
            try:
                raw = float(nx.degree_assortativity_coefficient(simple))
                assortativity = round(raw, 8) if math.isfinite(raw) else None
            except Exception:
                assortativity = None
        return {
            "available": True,
            "max_core": int(max_core),
            "degeneracy": int(max_core),
            "core_distribution": {
                str(key): value for key, value in sorted(counts.items())
            },
            "degree_assortativity": assortativity,
        }
    except Exception as exc:
        return {"available": False, "reason": type(exc).__name__}


def _spectral_summary(graph: Any, nx: Any, budget: dict[str, int]) -> dict[str, Any]:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n < 2:
        return {"available": False, "reason": "too_small"}
    if n > budget["spectral_nodes"] or m > budget["spectral_edges"]:
        return {
            "available": False,
            "reason": "resource_budget",
            "node_limit": budget["spectral_nodes"],
            "edge_limit": budget["spectral_edges"],
        }

    simple = nx.Graph(graph)
    simple.remove_edges_from(list(nx.selfloop_edges(simple)))
    if simple.number_of_edges() == 0:
        return {
            "available": True,
            "spectral_radius": 0.0,
            "algebraic_connectivity": 0.0,
            "connected": simple.number_of_nodes() <= 1,
        }

    try:
        adjacency = nx.to_scipy_sparse_array(simple, dtype=float, weight=None, format="csr")
        laplacian = nx.normalized_laplacian_matrix(simple, weight=None)
        if n <= 64:
            dense_a = np.asarray(adjacency.toarray(), dtype=float)
            dense_l = np.asarray(laplacian.toarray(), dtype=float)
            adjacency_values = np.linalg.eigvalsh(dense_a)
            laplacian_values = np.linalg.eigvalsh(dense_l)
            spectral_radius = float(np.max(np.abs(adjacency_values)))
            ordered = np.sort(np.real(laplacian_values))
            algebraic = float(ordered[1]) if ordered.size >= 2 else 0.0
            method = "dense_eigvalsh"
        else:
            from scipy.sparse.linalg import eigsh

            largest = eigsh(
                adjacency,
                k=1,
                which="LM",
                return_eigenvectors=False,
            )
            smallest = eigsh(
                laplacian,
                k=2,
                which="SM",
                return_eigenvectors=False,
            )
            spectral_radius = float(np.max(np.abs(largest)))
            ordered = np.sort(np.real(smallest))
            algebraic = float(ordered[1]) if ordered.size >= 2 else 0.0
            method = "sparse_eigsh"

        return {
            "available": True,
            "method": method,
            "spectral_radius": round(spectral_radius, 8),
            "algebraic_connectivity": round(max(0.0, algebraic), 10),
            "connected": bool(nx.is_connected(simple)),
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

    graph, source_filename = _coerce_graph(graph, nx)

    if not (
        callable(getattr(graph, "number_of_nodes", None))
        and callable(getattr(graph, "number_of_edges", None))
    ):
        raise TypeError("Expected a NetworkX-compatible graph object or supported graph file.")

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

    weight_inspection = _inspect_edge_weights(graph, attribute=weight or "weight")
    if weight is not None and weight_inspection.get("invalid_weights", 0):
        raise ValueError(
            "Dijkstra requires finite, non-negative weights for every weighted edge."
        )
    resolved_weight = weight or weight_inspection.get("selected")
    if n > budget["path_nodes"] or m > budget["path_edges"]:
        paths = {
            "available": False,
            "reason": "resource_budget",
            "node_limit": budget["path_nodes"],
            "edge_limit": budget["path_edges"],
            "weighted": bool(resolved_weight),
            "weight_attribute": resolved_weight,
        }
    else:
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
    core = _core_summary(graph, nx, budget)
    spectral = _spectral_summary(graph, nx, budget)

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

    if weight_inspection.get("invalid_weights", 0):
        findings.append(
            beacon(
                "graph.invalid_edge_weights",
                "Invalid edge weights prevent weighted shortest-path analysis",
                severity="high" if weight_inspection["negative_weights"] else "medium",
                summary=(
                    f"{weight_inspection['invalid_weights']:,} edges contain negative, "
                    "non-finite, or nonnumeric weights. Shortest paths use "
                    "unweighted BFS instead of Dijkstra."
                ),
                recommendation=(
                    "Validate edge costs and choose a weighted path algorithm "
                    "only after confirming its assumptions."
                ),
                evidence=dict(weight_inspection),
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

    algebraic = spectral.get("algebraic_connectivity")
    if (
        spectral.get("available")
        and spectral.get("connected")
        and n >= 10
        and isinstance(algebraic, (float, int))
        and float(algebraic) <= 1.0e-4
    ):
        findings.append(
            beacon(
                "graph.spectral_fragility",
                "The connected graph has extremely weak spectral connectivity",
                severity="medium",
                confidence=0.85,
                summary=(
                    "The normalized-Laplacian algebraic connectivity is "
                    f"{float(algebraic):.3g}, indicating a fragile bottlenecked topology."
                ),
                recommendation="Inspect sparse cuts, bridges, and communities that can split the network.",
                evidence=dict(spectral),
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
        "weight_validation": weight_inspection,
        "degree": degrees,
        "components": components,
        "shortest_paths": paths,
        "centrality": centrality,
        "cut_structure": cuts,
        "communities": communities,
        "clustering": clustering,
        "core": core,
        "spectral": spectral,
        "algorithm_budget": dict(budget),
    }

    return AnalysisResult(
        {
            "dataset_id": None,
            "filename": source_filename or type(graph).__name__,
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
