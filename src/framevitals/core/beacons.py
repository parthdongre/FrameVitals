"""Small common finding schema shared by structured-data analyzers."""

from __future__ import annotations

from typing import Any


def beacon(
    code: str,
    title: str,
    *,
    severity: str = "info",
    confidence: float = 1.0,
    summary: str | None = None,
    recommendation: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one protocol-facing Beacon/finding.

    The result intentionally remains a plain dictionary so it is compatible
    with the existing 0.x AnalysisResult/findings schema.
    """
    item: dict[str, Any] = {
        "code": str(code),
        "severity": str(severity),
        "title": str(title),
        "confidence": round(max(0.0, min(float(confidence), 1.0)), 4),
    }
    if summary:
        item["summary"] = str(summary)
    if recommendation:
        item["recommendation"] = str(recommendation)
    if evidence:
        item["evidence"] = dict(evidence)
    return item
