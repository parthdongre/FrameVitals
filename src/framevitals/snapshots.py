"""Compact versioned snapshots and lightweight local monitoring history."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_SCHEMA_VERSION = "1"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _structured_state(result: Mapping[str, Any], source_kind: str) -> dict[str, Any]:
    if source_kind == "graph":
        graph = _as_mapping(result.get("graph"))
        components = _as_mapping(graph.get("components"))
        degree = _as_mapping(graph.get("degree"))
        return {
            "nodes": graph.get("nodes"),
            "edges": graph.get("edges"),
            "density": graph.get("density"),
            "isolates": graph.get("isolates"),
            "self_loops": graph.get("self_loops"),
            "largest_component_ratio": components.get("largest_component_ratio"),
            "component_count": components.get("component_count"),
            "degree_mean": degree.get("mean"),
            "degree_p95": degree.get("p95"),
            "degree_max": degree.get("max"),
        }

    if source_kind == "tensor":
        tensor = _as_mapping(result.get("tensor"))
        matrix = _as_mapping(tensor.get("matrix"))
        distribution = _as_mapping(tensor.get("distribution"))
        return {
            "shape": list(tensor.get("shape", []) or []),
            "dtype": tensor.get("dtype"),
            "size": tensor.get("size"),
            "zero_fraction": tensor.get("zero_fraction_sample"),
            "near_zero_fraction": tensor.get("near_zero_fraction_sample"),
            "mean": distribution.get("mean"),
            "std": distribution.get("std"),
            "rank_ratio": matrix.get("rank_ratio"),
            "effective_rank": matrix.get("effective_rank"),
            "condition_number": matrix.get("condition_number"),
        }

    if source_kind == "model":
        model = _as_mapping(result.get("model"))
        runtime = _as_mapping(model.get("runtime"))
        runtime_summary = _as_mapping(runtime.get("summary"))
        optimizer = _as_mapping(model.get("optimizer"))
        return {
            "framework": model.get("framework"),
            "architecture": model.get("architecture"),
            "parameters": model.get("parameters"),
            "trainable_parameters": model.get("trainable_parameters"),
            "modules": model.get("modules"),
            "tensor_count": model.get("tensor_count"),
            "nodes": model.get("nodes"),
            "edges": model.get("edges"),
            "runtime": dict(runtime_summary),
            "optimizer": {
                "class_name": optimizer.get("class_name"),
                "group_count": optimizer.get("group_count"),
                "tracked_parameter_count": optimizer.get("tracked_parameter_count"),
                "untracked_trainable_parameters": list(
                    optimizer.get("untracked_trainable_parameters", []) or []
                ),
                "duplicate_parameter_assignments": list(
                    optimizer.get("duplicate_parameter_assignments", []) or []
                ),
                "learning_rates": list(optimizer.get("learning_rates", []) or []),
                "learning_rate_spread": optimizer.get("learning_rate_spread"),
                "estimated_state_bytes": optimizer.get("estimated_state_bytes"),
            } if optimizer else {},
        }

    if source_kind == "nested":
        nested = _as_mapping(result.get("nested"))
        arrays = _as_mapping(nested.get("arrays"))
        objects = _as_mapping(nested.get("objects"))
        return {
            "nodes_observed": nested.get("nodes_observed"),
            "max_depth": nested.get("max_depth"),
            "null_fraction": nested.get("null_fraction"),
            "path_type_conflict_count": nested.get("path_type_conflict_count"),
            "cyclic_references": nested.get("cyclic_references"),
            "array_max_length": arrays.get("max_length"),
            "unique_object_shapes": objects.get("unique_shapes"),
        }

    if source_kind == "relational":
        relational = _as_mapping(result.get("relational"))
        tables = _as_mapping(relational.get("tables"))
        return {
            "table_count": relational.get("table_count"),
            "relationship_count": relational.get("relationship_count"),
            "isolated_tables": list(relational.get("isolated_tables", []) or []),
            "tables": {
                str(name): {
                    "rows": _as_mapping(table).get("rows"),
                    "columns": _as_mapping(table).get("columns"),
                    "key_candidates": list(
                        _as_mapping(table).get("key_candidates", []) or []
                    ),
                }
                for name, table in tables.items()
            },
        }

    return {}


def _state_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    profile = _as_mapping(result.get("profile"))
    health = _as_mapping(result.get("health"))
    ml = _as_mapping(result.get("ml_readiness"))
    findings = result.get("findings", [])
    if not isinstance(findings, list):
        findings = []

    source_kind = str(result.get("source_kind") or "tabular")
    return {
        "result_schema_version": result.get("result_schema_version"),
        "analysis_mode": result.get("analysis_mode"),
        "source_kind": source_kind,
        "dataset": {
            "shape": dict(_as_mapping(profile.get("shape"))),
            "dtypes": dict(_as_mapping(profile.get("dtypes"))),
            "missing_percent": dict(_as_mapping(profile.get("missing_percent"))),
            "duplicate_percent": profile.get("duplicate_percent"),
            "memory_usage_mb": profile.get("memory_usage_mb"),
        },
        "structured": _structured_state(result, source_kind),
        "health": {
            "overall_score": health.get("overall_score"),
            "label": health.get("label"),
            "components": dict(_as_mapping(health.get("components"))),
        },
        "ml_readiness": {
            "score": ml.get("score"),
            "label": ml.get("label"),
            "issues": dict(_as_mapping(ml.get("issues"))),
        },
        "finding_codes": sorted(
            str(item.get("code"))
            for item in findings
            if isinstance(item, Mapping) and item.get("code")
        ),
        "config": dict(_as_mapping(result.get("config"))),
    }


def _fingerprint(state: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(state),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _created_at(snapshot: Mapping[str, Any]) -> datetime:
    value = snapshot.get("created_at")
    if not isinstance(value, str):
        raise ValueError("Snapshot is missing a valid created_at timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Snapshot has an invalid created_at timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_snapshot_payload(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported FrameVitals snapshot schema version: "
            f"{snapshot.get('snapshot_schema_version')!r}"
        )
    state = snapshot.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("Snapshot is missing a valid state object.")
    _created_at(snapshot)

    fingerprint = snapshot.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Snapshot is missing a valid fingerprint.")
    expected = _fingerprint(state)
    if fingerprint != expected:
        raise ValueError(
            "Snapshot fingerprint does not match its stored state; "
            "the snapshot may be corrupted or modified."
        )


def _safe_label(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned[:80] or None


class AnalysisSnapshot(dict):
    """Small JSON-friendly state record for monitoring and history."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_json(
        self,
        destination: str | Path | None = None,
        *,
        indent: int = 2,
    ) -> str | Path:
        rendered = json.dumps(dict(self), indent=indent, default=str)
        if destination is None:
            return rendered
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        return path

    def diff(self, other: Mapping[str, Any]) -> dict[str, Any]:
        return compare_snapshots(self, other)


def create_snapshot(result: Mapping[str, Any]) -> AnalysisSnapshot:
    """Create a deterministic compact state snapshot from an analysis result."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")
    state = _state_payload(result)
    return AnalysisSnapshot({
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": result.get("dataset_id"),
            "filename": result.get("filename"),
            "kind": result.get("source_kind", "tabular"),
        },
        "fingerprint": _fingerprint(state),
        "state": state,
    })


def load_snapshot(path: str | Path) -> AnalysisSnapshot:
    """Load and validate a FrameVitals snapshot from JSON."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Snapshot not found: {source}")
    if not source.is_file():
        raise ValueError(f"Expected a snapshot file, got: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot is not valid JSON: {source}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Snapshot JSON must contain an object.")
    _validate_snapshot_payload(payload)
    return AnalysisSnapshot(payload)


def compare_snapshots(
    reference: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare compact analysis snapshots without requiring raw datasets."""
    ref_state = _as_mapping(reference.get("state"))
    cur_state = _as_mapping(current.get("state"))
    ref_dataset = _as_mapping(ref_state.get("dataset"))
    cur_dataset = _as_mapping(cur_state.get("dataset"))

    ref_dtypes = dict(_as_mapping(ref_dataset.get("dtypes")))
    cur_dtypes = dict(_as_mapping(cur_dataset.get("dtypes")))
    ref_columns = set(ref_dtypes)
    cur_columns = set(cur_dtypes)

    type_changes = {
        column: {"reference": ref_dtypes[column], "current": cur_dtypes[column]}
        for column in sorted(ref_columns & cur_columns)
        if ref_dtypes[column] != cur_dtypes[column]
    }

    ref_missing = dict(_as_mapping(ref_dataset.get("missing_percent")))
    cur_missing = dict(_as_mapping(cur_dataset.get("missing_percent")))
    missing_changes: dict[str, dict[str, float]] = {}
    for column in sorted(set(ref_missing) & set(cur_missing)):
        before = _number(ref_missing[column])
        after = _number(cur_missing[column])
        if before is None or after is None or before == after:
            continue
        missing_changes[column] = {
            "reference": round(before, 4),
            "current": round(after, 4),
            "delta": round(after - before, 4),
        }

    ref_health = _number(_as_mapping(ref_state.get("health")).get("overall_score"))
    cur_health = _number(_as_mapping(cur_state.get("health")).get("overall_score"))
    ref_ml = _number(_as_mapping(ref_state.get("ml_readiness")).get("score"))
    cur_ml = _number(_as_mapping(cur_state.get("ml_readiness")).get("score"))

    ref_findings = set(ref_state.get("finding_codes", []) or [])
    cur_findings = set(cur_state.get("finding_codes", []) or [])

    ref_kind = str(ref_state.get("source_kind") or "tabular")
    cur_kind = str(cur_state.get("source_kind") or "tabular")
    ref_structured = dict(_as_mapping(ref_state.get("structured")))
    cur_structured = dict(_as_mapping(cur_state.get("structured")))

    changed = bool(reference.get("fingerprint") != current.get("fingerprint"))
    return {
        "changed": changed,
        "reference_fingerprint": reference.get("fingerprint"),
        "current_fingerprint": current.get("fingerprint"),
        "schema": {
            "added_columns": sorted(cur_columns - ref_columns),
            "removed_columns": sorted(ref_columns - cur_columns),
            "type_changes": type_changes,
        },
        "missingness_changes": missing_changes,
        "health_delta": (
            round(cur_health - ref_health, 4)
            if ref_health is not None and cur_health is not None
            else None
        ),
        "ml_readiness_delta": (
            round(cur_ml - ref_ml, 4)
            if ref_ml is not None and cur_ml is not None
            else None
        ),
        "source_kind": {
            "reference": ref_kind,
            "current": cur_kind,
            "changed": ref_kind != cur_kind,
        },
        "structured": {
            "changed": ref_structured != cur_structured,
            "reference": ref_structured,
            "current": cur_structured,
        },
        "findings": {
            "new": sorted(cur_findings - ref_findings),
            "resolved": sorted(ref_findings - cur_findings),
        },
    }


class SnapshotHistory:
    """Filesystem-backed history of compact FrameVitals snapshots."""

    def __init__(self, directory: str | Path = ".framevitals/history") -> None:
        self.directory = Path(directory)

    def __len__(self) -> int:
        return len(self.paths())

    def paths(self) -> list[Path]:
        """Return snapshot files in chronological filename order."""
        if not self.directory.exists():
            return []
        return sorted(path for path in self.directory.glob("*.json") if path.is_file())

    def snapshots(self) -> list[AnalysisSnapshot]:
        """Load all valid snapshots ordered by their created-at timestamp."""
        items = [load_snapshot(path) for path in self.paths()]
        items.sort(key=_created_at)
        return items

    def add(
        self,
        result_or_snapshot: Mapping[str, Any],
        *,
        label: str | None = None,
    ) -> Path:
        """Persist an analysis result or an existing snapshot and return its path."""
        if not isinstance(result_or_snapshot, Mapping):
            raise TypeError("result_or_snapshot must be a mapping.")

        if result_or_snapshot.get("snapshot_schema_version") is not None:
            _validate_snapshot_payload(result_or_snapshot)
            snapshot = AnalysisSnapshot(dict(result_or_snapshot))
            created_at = _created_at(snapshot)
        else:
            snapshot = create_snapshot(result_or_snapshot)
            created_at = _created_at(snapshot)

        self.directory.mkdir(parents=True, exist_ok=True)
        timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        fingerprint = str(snapshot.get("fingerprint") or "unknown")[:12]
        label_part = _safe_label(label)
        filename = "_".join(
            part for part in (timestamp, label_part, fingerprint) if part
        ) + ".json"
        path = self.directory / filename
        snapshot.to_json(path)
        return path

    def latest(self) -> AnalysisSnapshot | None:
        """Return the newest snapshot or ``None`` for an empty history."""
        items = self.snapshots()
        return items[-1] if items else None

    def previous(self) -> AnalysisSnapshot | None:
        """Return the snapshot immediately before the latest one, if available."""
        items = self.snapshots()
        return items[-2] if len(items) >= 2 else None

    def compare_latest(self) -> dict[str, Any]:
        """Compare the two newest snapshots."""
        items = self.snapshots()
        if len(items) < 2:
            raise ValueError("Snapshot history needs at least two entries to compare.")
        return compare_snapshots(items[-2], items[-1])

    def timeline(self) -> list[dict[str, Any]]:
        """Return compact chronological monitoring points for charts or logs."""
        rows: list[dict[str, Any]] = []
        for snapshot in self.snapshots():
            state = _as_mapping(snapshot.get("state"))
            dataset = _as_mapping(state.get("dataset"))
            health = _as_mapping(state.get("health"))
            ml = _as_mapping(state.get("ml_readiness"))
            findings = state.get("finding_codes", [])
            if not isinstance(findings, list):
                findings = []
            rows.append({
                "created_at": snapshot.get("created_at"),
                "fingerprint": snapshot.get("fingerprint"),
                "filename": _as_mapping(snapshot.get("source")).get("filename"),
                "source_kind": state.get("source_kind", "tabular"),
                "shape": dict(_as_mapping(dataset.get("shape"))),
                "structured": dict(_as_mapping(state.get("structured"))),
                "health_score": _number(health.get("overall_score")),
                "ml_readiness_score": _number(ml.get("score")),
                "finding_count": len(findings),
            })
        return rows
