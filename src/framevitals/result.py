"""Public result objects for FrameVitals.

``AnalysisResult`` intentionally subclasses :class:`dict` during the 0.x series.
That preserves existing mapping behaviour and JSON compatibility while adding a
more discoverable object-oriented API for notebooks, applications, and reports.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from framevitals.findings import findings_from_signals, recommendations_from_findings


class ColumnResult(dict):
    """Structured view of one analyzed column."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class AnalysisResult(dict):
    """Backward-compatible result returned by :func:`framevitals.analyze`."""

    schema_version = "1"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setdefault("result_schema_version", self.schema_version)
        if "findings" not in self:
            signals = self.get("signals", [])
            if not isinstance(signals, list):
                signals = []
            self["findings"] = findings_from_signals(signals)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def findings(self) -> list[dict[str, Any]]:
        value = self.get("findings", [])
        return value if isinstance(value, list) else []

    @property
    def recommendations(self) -> list[str]:
        return recommendations_from_findings(self.findings)

    @property
    def health(self) -> dict[str, Any]:
        value = self.get("health", {})
        return value if isinstance(value, dict) else {}

    @property
    def ml_readiness(self) -> dict[str, Any]:
        value = self.get("ml_readiness", {})
        return value if isinstance(value, dict) else {}

    @property
    def shape(self) -> dict[str, Any]:
        profile = self.get("profile", {})
        if not isinstance(profile, dict):
            return {}
        shape = profile.get("shape", {})
        return shape if isinstance(shape, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain-dictionary copy of the complete result."""
        return deepcopy(dict(self))

    def summary(self) -> dict[str, Any]:
        """Return a concise, stable high-level summary of the analysis."""
        health = self.health
        ml_readiness = self.ml_readiness
        timings = self.get("timings_ms", {})
        if not isinstance(timings, dict):
            timings = {}

        severity_counts: dict[str, int] = {}
        for finding in self.findings:
            severity = str(finding.get("severity") or "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "dataset_id": self.get("dataset_id"),
            "filename": self.get("filename"),
            "analysis_mode": self.get("analysis_mode"),
            "result_schema_version": self.get("result_schema_version"),
            "shape": dict(self.shape),
            "health": {
                "overall_score": health.get("overall_score"),
                "label": health.get("label"),
            },
            "ml_readiness": {
                "score": ml_readiness.get("score"),
                "label": ml_readiness.get("label"),
            },
            "finding_count": len(self.findings),
            "finding_severity_counts": severity_counts,
            "artifacts_enabled": bool(self.get("artifacts_enabled", False)),
            "total_ms": timings.get("total"),
        }

    def column(self, name: str) -> ColumnResult:
        """Return the combined profile, role, and semantic view for one column."""
        profile = self.get("profile", {})
        roles = self.get("column_roles", {})
        if not isinstance(profile, dict):
            profile = {}
        if not isinstance(roles, dict):
            roles = {}

        columns = profile.get("columns", [])
        if name not in columns and name not in roles:
            raise KeyError(f"Column not found in analysis result: {name}")

        role_info = roles.get(name, {})
        if not isinstance(role_info, dict):
            role_info = {}

        numeric_summary = profile.get("numeric_summary", {})
        categorical_summary = profile.get("categorical_summary", {})
        correlations = profile.get("correlations", {})
        if not isinstance(numeric_summary, dict):
            numeric_summary = {}
        if not isinstance(categorical_summary, dict):
            categorical_summary = {}
        if not isinstance(correlations, dict):
            correlations = {}

        payload = {
            "name": name,
            "dtype": profile.get("dtypes", {}).get(name),
            "roles": list(role_info.get("roles", [])),
            "semantic_type": role_info.get("semantic_type"),
            "semantic_candidates": list(role_info.get("semantic_candidates", [])),
            "semantic_sample_size": role_info.get("semantic_sample_size", 0),
            "missing_count": profile.get("missing_counts", {}).get(name),
            "missing_percent": profile.get("missing_percent", {}).get(name),
            "unique_count": role_info.get("unique_count"),
            "unique_ratio": role_info.get("unique_ratio"),
            "non_missing_count": role_info.get("non_missing_count"),
            "is_numeric": role_info.get("is_numeric"),
            "is_categorical": role_info.get("is_categorical"),
            "numeric_summary": numeric_summary.get(name),
            "categorical_summary": categorical_summary.get(name),
            "correlations": correlations.get(name, {}),
        }
        return ColumnResult(payload)

    def to_json(
        self,
        destination: str | Path | None = None,
        *,
        indent: int = 2,
    ) -> str | Path:
        """Serialize the complete result to JSON or write it to ``destination``."""
        rendered = json.dumps(self.to_dict(), indent=indent, default=str)
        if destination is None:
            return rendered

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        return path

    def summary_text(self) -> str:
        """Render a compact human-readable terminal summary."""
        from framevitals.reporting.terminal import render_terminal_summary

        return render_terminal_summary(self)

    def to_html(self, destination: str | Path | None = None) -> str | Path:
        """Render a self-contained HTML report and optionally write it to disk."""
        from framevitals.reporting.html import render_html_report

        rendered = render_html_report(self)
        if destination is None:
            return rendered

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return path

    def snapshot(self, destination: str | Path | None = None):
        """Create a compact versioned monitoring snapshot from this result."""
        from framevitals.snapshots import create_snapshot

        snapshot = create_snapshot(self)
        if destination is not None:
            snapshot.to_json(destination)
        return snapshot

    def _repr_html_(self) -> str:
        """Provide a compact rich representation in Jupyter-compatible clients."""
        from framevitals.reporting.html import render_notebook_summary

        return render_notebook_summary(self)
