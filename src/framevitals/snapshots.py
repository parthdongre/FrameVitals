"""Compact versioned snapshots derived from FrameVitals analysis results."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_SCHEMA_VERSION = "1"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _state_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    profile = _as_mapping(result.get("profile"))
    health = _as_mapping(result.get("health"))
    ml = _as_mapping(result.get("ml_readiness"))
    findings = result.get("findings", [])
    if not isinstance(findings, list):
        findings = []

    return {
        "result_schema_version": result.get("result_schema_version"),
        "analysis_mode": result.get("analysis_mode"),
        "dataset": {
            "shape": dict(_as_mapping(profile.get("shape"))),
            "dtypes": dict(_as_mapping(profile.get("dtypes"))),
            "missing_percent": dict(_as_mapping(profile.get("missing_percent"))),
            "duplicate_percent": profile.get("duplicate_percent"),
            "memory_usage_mb": profile.get("memory_usage_mb"),
        },
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
    state = _state_payload(result)
    return AnalysisSnapshot({
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_id": result.get("dataset_id"),
            "filename": result.get("filename"),
        },
        "fingerprint": _fingerprint(state),
        "state": state,
    })


def load_snapshot(path: str | Path) -> AnalysisSnapshot:
    """Load and validate a FrameVitals snapshot from JSON."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Snapshot not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot is not valid JSON: {source}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Snapshot JSON must contain an object.")
    if payload.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported FrameVitals snapshot schema version: "
            f"{payload.get('snapshot_schema_version')!r}"
        )
    if not isinstance(payload.get("state"), dict):
        raise ValueError("Snapshot is missing a valid state object.")
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

    changed = bool(
        reference.get("fingerprint") != current.get("fingerprint")
    )
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
        "findings": {
            "new": sorted(cur_findings - ref_findings),
            "resolved": sorted(ref_findings - cur_findings),
        },
    }
