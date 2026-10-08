"""Bounded diagnostics for nested Python/JSON-like structures."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from framevitals.core.beacons import beacon
from framevitals.quality_results import DriftResult
from framevitals.result import AnalysisResult


def _type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    return type(value).__name__


def inspect_nested(
    data: Any,
    *,
    max_nodes: int = 100_000,
    max_depth: int = 32,
) -> dict[str, Any]:
    stack: list[tuple[str, Any, int]] = [("$", data, 0)]
    visited_ids: set[int] = set()
    nodes = 0
    max_observed_depth = 0
    truncated = False

    type_counts: Counter[str] = Counter()
    path_type_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    object_key_counts: Counter[str] = Counter()
    object_shapes: Counter[tuple[str, ...]] = Counter()
    array_lengths: list[int] = []
    empty_arrays = 0
    empty_objects = 0
    null_values = 0
    scalar_values = 0
    cyclic_references = 0

    while stack:
        path, value, depth = stack.pop()
        if nodes >= max_nodes:
            truncated = True
            break

        nodes += 1
        max_observed_depth = max(max_observed_depth, depth)
        label = _type_label(value)
        type_counts[label] += 1
        path_type_counts[path][label] += 1

        if value is None:
            null_values += 1
            continue

        if isinstance(value, dict):
            identity = id(value)
            if identity in visited_ids:
                cyclic_references += 1
                continue
            visited_ids.add(identity)

            keys = tuple(sorted(str(key) for key in value.keys()))
            object_shapes[keys] += 1
            if not keys:
                empty_objects += 1

            for key, child in value.items():
                key_text = str(key)
                object_key_counts[key_text] += 1
                if depth < max_depth:
                    stack.append((f"{path}.{key_text}", child, depth + 1))
                else:
                    truncated = True
            continue

        if isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in visited_ids:
                cyclic_references += 1
                continue
            visited_ids.add(identity)

            length = len(value)
            array_lengths.append(length)
            if length == 0:
                empty_arrays += 1
            if depth < max_depth:
                for child in value:
                    # Arrays describe repeated values/records. Use one wildcard
                    # path regardless of array length so schema/type conflicts
                    # are compared across elements instead of hidden by index.
                    stack.append((f"{path}[*]", child, depth + 1))
            else:
                truncated = True
            continue

        scalar_values += 1

    path_conflicts = []
    for path, counts in sorted(path_type_counts.items()):
        observed = {name: count for name, count in counts.items() if count > 0}
        non_null_types = [name for name in observed if name != "null"]
        if len(non_null_types) > 1:
            path_conflicts.append({
                "path": path,
                "types": observed,
            })

    array_summary = {
        "count": len(array_lengths),
        "empty": empty_arrays,
        "mean_length": round(float(np.mean(array_lengths)), 6) if array_lengths else 0.0,
        "median_length": round(float(np.median(array_lengths)), 6) if array_lengths else 0.0,
        "p95_length": round(float(np.quantile(array_lengths, 0.95)), 6) if array_lengths else 0.0,
        "max_length": max(array_lengths, default=0),
    }

    common_shapes = [
        {"keys": list(keys), "count": count}
        for keys, count in object_shapes.most_common(20)
    ]

    return {
        "nodes_observed": nodes,
        "max_depth": max_observed_depth,
        "truncated": truncated,
        "type_counts": dict(type_counts),
        "null_values": null_values,
        "scalar_values": scalar_values,
        "null_fraction": round(null_values / max(1, nodes), 8),
        "cyclic_references": cyclic_references,
        "arrays": array_summary,
        "objects": {
            "count": int(type_counts.get("object", 0)),
            "empty": empty_objects,
            "unique_shapes": len(object_shapes),
            "common_shapes": common_shapes,
            "top_keys": [
                {"key": key, "count": count}
                for key, count in object_key_counts.most_common(30)
            ],
        },
        "path_type_conflicts": path_conflicts[:100],
        "path_type_conflict_count": len(path_conflicts),
    }


def _health_label(score: float) -> str:
    if score >= 90:
        return "healthy"
    if score >= 75:
        return "good"
    if score >= 55:
        return "attention"
    return "critical"


def analyze_nested(data: Any, *, depth: str | None = None) -> AnalysisResult:
    mode = str(depth or "standard").lower()
    budgets = {
        "quick": (20_000, 16),
        "standard": (100_000, 32),
        "deep": (300_000, 48),
        "research": (1_000_000, 64),
    }
    max_nodes, max_depth = budgets.get(mode, budgets["standard"])
    summary = inspect_nested(data, max_nodes=max_nodes, max_depth=max_depth)

    findings: list[dict[str, Any]] = []

    if summary["path_type_conflict_count"]:
        findings.append(beacon(
            "nested.type_conflicts",
            "Nested paths contain conflicting value types",
            severity="high",
            confidence=1.0,
            summary=f"{summary['path_type_conflict_count']} paths contain multiple non-null types.",
            recommendation="Normalize inconsistent field types before downstream parsing or modeling.",
            evidence={"paths": summary["path_type_conflicts"][:30]},
        ))

    if summary["cyclic_references"]:
        findings.append(beacon(
            "nested.cycles",
            "Cyclic Python references were detected",
            severity="medium",
            confidence=1.0,
            summary=f"{summary['cyclic_references']} cyclic references were encountered.",
            recommendation="Remove cycles before serializing this structure as JSON-like data.",
        ))

    if summary["max_depth"] >= max_depth:
        findings.append(beacon(
            "nested.deep_structure",
            "The nested structure reaches the configured depth bound",
            severity="medium",
            confidence=0.95,
            summary=f"Observed nesting reached depth {summary['max_depth']}.",
            recommendation="Review extreme nesting and consider flattening repeated structural layers.",
        ))

    if summary["arrays"]["max_length"] >= 10_000:
        findings.append(beacon(
            "nested.large_array",
            "Very large nested arrays were detected",
            severity="medium",
            confidence=1.0,
            summary=f"Largest observed array contains {summary['arrays']['max_length']:,} entries.",
            recommendation="Consider streaming or columnar representation for large repeated arrays.",
        ))

    score = 100.0
    score -= min(35.0, summary["path_type_conflict_count"] * 5.0)
    score -= min(20.0, summary["cyclic_references"] * 5.0)
    if summary["truncated"]:
        score -= 5.0
    score = round(max(0.0, min(100.0, score)), 2)

    nested_summary = {
        **summary,
        "algorithm_budget": {"max_nodes": max_nodes, "max_depth": max_depth},
    }

    return AnalysisResult({
        "dataset_id": None,
        "filename": type(data).__name__,
        "analysis_mode": mode,
        "source_kind": "nested",
        "profile": {
            "shape": {
                "rows": int(summary["nodes_observed"]),
                "columns": int(summary["objects"]["unique_shapes"]),
            },
            "structure": nested_summary,
        },
        "nested": nested_summary,
        "health": {"overall_score": score, "label": _health_label(score)},
        "ml_readiness": {"score": None, "label": "not_applicable"},
        "findings": findings,
        "artifacts_enabled": False,
        "execution": {
            "method": "bounded_nested_diagnostics",
            "sampled": bool(summary["truncated"]),
            "resource_bounded": True,
        },
    })


def compare_nested(reference: Any, current: Any) -> DriftResult:
    ref = inspect_nested(reference, max_nodes=100_000, max_depth=32)
    cur = inspect_nested(current, max_nodes=100_000, max_depth=32)

    ref_types = Counter(ref["type_counts"])
    cur_types = Counter(cur["type_counts"])
    all_types = sorted(set(ref_types) | set(cur_types))
    type_deltas = {
        name: {"reference": int(ref_types.get(name, 0)), "current": int(cur_types.get(name, 0))}
        for name in all_types
        if ref_types.get(name, 0) != cur_types.get(name, 0)
    }

    ref_keys = {item["key"]: item["count"] for item in ref["objects"]["top_keys"]}
    cur_keys = {item["key"]: item["count"] for item in cur["objects"]["top_keys"]}
    ref_key_set = set(ref_keys)
    cur_key_set = set(cur_keys)

    node_change = abs(cur["nodes_observed"] - ref["nodes_observed"]) / max(1, ref["nodes_observed"])
    depth_change = abs(cur["max_depth"] - ref["max_depth"]) / max(1, ref["max_depth"])
    conflict_change = abs(
        cur["path_type_conflict_count"] - ref["path_type_conflict_count"]
    ) / max(1, ref["path_type_conflict_count"] or 1)
    key_churn = (
        len(ref_key_set ^ cur_key_set) / max(1, len(ref_key_set | cur_key_set))
    )

    score = max(
        min(1.0, node_change),
        min(1.0, depth_change),
        min(1.0, conflict_change),
        min(1.0, key_churn * 2.0),
    )
    severity = "severe" if score >= 0.75 else "moderate" if score >= 0.40 else "minor" if score >= 0.15 else "stable"
    status = "fail" if severity == "severe" else "warn" if severity != "stable" else "pass"

    return DriftResult({
        "available": True,
        "source_kind": "nested",
        "gate": {"status": status, "severity": severity},
        "summary": {
            "overall_verdict": severity,
            "change_score": round(score, 6),
        },
        "nested": {
            "reference_nodes": ref["nodes_observed"],
            "current_nodes": cur["nodes_observed"],
            "reference_depth": ref["max_depth"],
            "current_depth": cur["max_depth"],
            "reference_type_conflicts": ref["path_type_conflict_count"],
            "current_type_conflicts": cur["path_type_conflict_count"],
            "added_keys": sorted(cur_key_set - ref_key_set),
            "removed_keys": sorted(ref_key_set - cur_key_set),
            "type_count_changes": type_deltas,
        },
        "columns": [],
    })
