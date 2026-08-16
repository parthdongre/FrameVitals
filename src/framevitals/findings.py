"""Normalized findings for FrameVitals analysis results.

The current pipeline already produces display-oriented signals with evidence and
recommendations.  This module converts those signals into a stable, machine-
readable finding shape without changing any analysis thresholds.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


_SEVERITY_MAP = {
    "high": "high",
    "critical": "critical",
    "medium": "medium",
    "moderate": "medium",
    "low": "low",
    "informational": "info",
    "info": "info",
    "none": "none",
}

_SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    "none": 5,
}


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "finding"


def normalize_severity(value: Any) -> str:
    """Return a stable lower-case severity label."""
    if value is None:
        return "none"
    text = str(value).strip().lower()
    return _SEVERITY_MAP.get(text, text or "none")


def _is_actionable_signal(signal: dict[str, Any]) -> bool:
    """Return whether an existing display signal represents an issue/review.

    Positive informational signals such as "Temporal Data: Detected" remain in
    ``signals`` but do not become findings.  ML readiness is included whenever
    it is not fully Ready because its existing signal does not use the generic
    ``Review`` status.
    """
    status = str(signal.get("status", "")).strip().lower()
    name = str(signal.get("name", "")).strip().lower()

    if status == "review":
        return True
    if name == "ml readiness" and status not in {"ready", "good"}:
        return True
    return False


def findings_from_signals(signals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize existing pipeline signals into actionable findings.

    No diagnostic logic is introduced here: the function only translates
    already-computed signal status/evidence/recommendations into a consistent
    schema for library consumers and report renderers.
    """
    findings: list[dict[str, Any]] = []

    for signal in signals:
        if not isinstance(signal, dict) or not _is_actionable_signal(signal):
            continue

        name = str(signal.get("name") or "Finding")
        severity = normalize_severity(signal.get("severity"))
        finding = {
            "code": f"signal.{_slug(name)}",
            "title": name,
            "severity": severity,
            "scope": "dataset",
            "status": signal.get("status"),
            "evidence": signal.get("evidence") or "",
            "recommendation": signal.get("recommendation") or "",
            "method": "signal_engine",
            "confidence": "deterministic",
        }
        findings.append(finding)

    findings.sort(
        key=lambda item: (
            _SEVERITY_ORDER.get(str(item.get("severity")), 99),
            str(item.get("code", "")),
        )
    )
    return findings


def recommendations_from_findings(
    findings: Iterable[dict[str, Any]],
) -> list[str]:
    """Return de-duplicated actionable recommendations in finding order."""
    recommendations: list[str] = []
    seen: set[str] = set()

    for finding in findings:
        recommendation = str(finding.get("recommendation") or "").strip()
        if not recommendation or recommendation in seen:
            continue
        seen.add(recommendation)
        recommendations.append(recommendation)

    return recommendations
