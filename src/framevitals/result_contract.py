"""Additive v1 result envelope. Never infer success, exactness, or zero from absence."""

from __future__ import annotations

from typing import Any

from framevitals.serialization import json_safe
from framevitals.scientific import score_interpretation

STATUSES = ("completed", "skipped", "unavailable", "not_applicable", "failed", "partial")


def _diagnostic(name: str, value: Any) -> dict[str, Any]:
    info = value if isinstance(value, dict) else {}
    raw = str(info.get("status") or "")
    reason = info.get("reason") or info.get("error")
    if raw in {"failed", "error"} or info.get("error"):
        status = "failed"
    elif raw.startswith("skipped") or raw in {"disabled", "deferred"}:
        status = "skipped"
    elif (
        info.get("label") == "not_applicable"
        or raw == "not_applicable"
        or reason in {"not_applicable", "not_a_matrix"}
    ):
        status = "not_applicable"
    elif info.get("available") is False:
        status = "unavailable"
    elif value is None or info.get("available") is None and not info:
        status = "unavailable"
    elif info.get("truncated") or info.get("partial"):
        status = "partial"
    else:
        status = "completed"
    sampled = info.get("sampled")
    approximate = info.get("approximate")
    return {
        "name": name,
        "status": status,
        "reason": reason,
        "method": info.get("method"),
        "coverage": info.get("coverage") or ("sampled" if sampled is True else "full" if sampled is False else "unknown"),
        "accuracy": "approximate"
        if approximate is True
        else "exact"
        if approximate is False
        else "unspecified",
    }


def build_envelope(
    result: dict[str, Any], *, source: dict[str, Any] | None = None, elapsed_ms: float | None = None
) -> dict[str, Any]:
    """Wrap either Prism or AnalysisResult, preserving its complete original payload."""
    analysis = result.get("analysis", result)
    execution = analysis.get("execution") or {}
    kind = analysis.get("source_kind", "tabular")
    diagnostics = []
    for name in (
        "profile",
        "health",
        "ml_readiness",
        "quality_diagnostics",
        "deep_statistics_v2",
        "anomalies_v2",
        "target_intelligence",
        "model_leaderboard",
        "explainability",
        "time_series",
        "text_profile",
        "graph",
        "tensor",
        "nested",
        "document",
        "model",
        "relational",
    ):
        if name in analysis:
            diagnostics.append(_diagnostic(name, analysis[name]))
    aliases = {
        "anomaly_detection": "anomalies_v2",
        "deep_statistics": "deep_statistics_v2",
        "modeling": "model_leaderboard",
    }
    for name, state in execution.get("module_status", {}).items():
        mapped = aliases.get(name, name)
        row = _diagnostic(mapped, {"status": state})
        if state == "ran":
            continue
        diagnostics = [d for d in diagnostics if d["name"] != mapped]
        diagnostics.append(row)
    for name in ("graph", "tensor", "model"):
        section = analysis.get(name)
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, dict) and any(
                    k in value
                    for k in ("available", "method", "approximate", "sampled", "error", "status")
                ):
                    diagnostics.append(_diagnostic(f"{name}.{key}", value))
    status = (
        "partial" if any(d["status"] in {"failed", "partial"} for d in diagnostics) else "completed"
    )
    return json_safe(
        {
            "schema_version": "1",
            "status": status,
            "source": source or {"kind": kind},
            "elapsed_ms": elapsed_ms,
            "execution": {
                **execution,
                "coverage": "sampled"
                if execution.get("sampled") is True
                else "full"
                if execution.get("sampled") is False
                else "unknown",
            },
            "scientific_score": score_interpretation(analysis),
            "scientific_methods_version": "2",
            "score_interpretation": "Heuristic diagnostic score; not calibrated risk or scientific certification.",
            "diagnostics": diagnostics,
            "findings": analysis.get("findings", []),
            "analysis": analysis,
            "change": result.get("change"),
            "trust": result.get("trust"),
        }
    )
