"""Bounded visualization data, without new analytical calculations."""

from __future__ import annotations

from itertools import islice
from typing import Any


def graph_excerpt(graph: Any) -> dict[str, Any]:
    nodes = list(islice(graph.nodes, 48))
    indexes = {node: index for index, node in enumerate(nodes)}
    edges = []
    scanned = 0
    for left, right in islice(graph.edges(), 5000):
        scanned += 1
        if left in indexes and right in indexes:
            edges.append([indexes[left], indexes[right]])
        if len(edges) >= 120:
            break
    return {
        "nodes": [{"id": i, "label": str(node)[:80]} for node, i in indexes.items()],
        "edges": edges,
        "directed": graph.is_directed(),
        "sampled": len(nodes) < graph.number_of_nodes() or scanned < graph.number_of_edges(),
        "layout": "circular_display_only",
        "edge_scan_limit": 5000,
    }
