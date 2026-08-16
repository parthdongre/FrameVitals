"""Normalized findings for FrameVitals analysis results.

FrameVitals diagnostics originate in several deterministic analysis layers. This
module translates their already-computed warnings/signals into one stable,
machine-readable finding shape without changing diagnostic thresholds.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


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


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings.sort(
        key=lambda item: (
            _SEVERITY_ORDER.get(str(item.get("severity")), 99),
            str(item.get("code", "")),
        )
    )
    return findings


def _is_actionable_signal(signal: dict[str, Any]) -> bool:
    status = str(signal.get("status", "")).strip().lower()
    name = str(signal.get("name", "")).strip().lower()

    if status == "review":
        return True
    if name == "ml readiness" and status not in {"ready", "good"}:
        return True
    return False


def findings_from_signals(signals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize existing display signals into actionable findings."""
    findings: list[dict[str, Any]] = []

    for signal in signals:
        if not isinstance(signal, dict) or not _is_actionable_signal(signal):
            continue

        name = str(signal.get("name") or "Finding")
        findings.append({
            "code": f"signal.{_slug(name)}",
            "title": name,
            "severity": normalize_severity(signal.get("severity")),
            "scope": "dataset",
            "status": signal.get("status"),
            "evidence": signal.get("evidence") or "",
            "recommendation": signal.get("recommendation") or "",
            "method": "signal_engine",
            "confidence": "deterministic",
        })

    return _sort_findings(findings)


def findings_from_target_intelligence(
    target_intelligence: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Normalize target-quality/leakage warnings into standard findings."""
    if not isinstance(target_intelligence, Mapping) or not target_intelligence.get("available"):
        return []

    target = str(target_intelligence.get("target_column") or "target")
    warnings = target_intelligence.get("warnings", [])
    if not isinstance(warnings, list):
        return []

    findings: list[dict[str, Any]] = []
    for warning in warnings:
        if not isinstance(warning, Mapping):
            continue
        code = str(warning.get("code") or "target.warning")
        evidence = str(warning.get("message") or "Target review is recommended.")

        if code == "target.id_like":
            title = "Identifier-like target"
            recommendation = (
                f"Verify that '{target}' represents an outcome rather than a record identifier "
                "before training a supervised model."
            )
        elif code == "target.high_missingness":
            title = "High target missingness"
            recommendation = (
                "Resolve or explicitly exclude rows with missing target labels before model training."
            )
        elif code == "target.high_cardinality_classification":
            title = "High-cardinality classification target"
            recommendation = (
                "Confirm that the target classes are intentional and have enough examples for reliable evaluation."
            )
        elif code.startswith("target.leakage."):
            feature = code.removeprefix("target.leakage.")
            title = f"Potential target leakage: {feature}"
            recommendation = (
                f"Review or remove '{feature}' before training if it contains information that would "
                "not be available at prediction time."
            )
        else:
            title = "Target review"
            recommendation = "Review the selected target and its relationship to the input features."

        findings.append({
            "code": code,
            "title": title,
            "severity": normalize_severity(warning.get("severity")),
            "scope": "target",
            "status": "Review",
            "evidence": evidence,
            "recommendation": recommendation,
            "method": "target_intelligence",
            "confidence": "deterministic",
        })

    return _sort_findings(findings)


def merge_findings(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge finding groups, de-duplicating by stable code."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for finding in group:
            code = str(finding.get("code") or "")
            if code and code in seen:
                continue
            if code:
                seen.add(code)
            merged.append(dict(finding))
    return _sort_findings(merged)


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
