"""Dict-compatible public results for checks, validation, drift, and quality gates."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class _QualityResult(dict):
    """Small mapping-compatible base with export conveniences."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self))

    def to_json(
        self,
        destination: str | Path | None = None,
        *,
        indent: int = 2,
    ) -> str | Path:
        rendered = json.dumps(self.to_dict(), indent=indent, default=str)
        if destination is None:
            return rendered
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        return path


class CheckResult(_QualityResult):
    """Dict-compatible result returned by :func:`framevitals.run_checks`."""

    @property
    def status(self) -> str:
        return str(self.get("status", "unknown"))

    @property
    def passed(self) -> bool:
        return bool(self.get("passed", False))

    @property
    def findings(self) -> list[dict[str, Any]]:
        value = self.get("findings", [])
        return value if isinstance(value, list) else []

    @property
    def results(self) -> list[dict[str, Any]]:
        value = self.get("results", [])
        return value if isinstance(value, list) else []

    def summary_text(self) -> str:
        summary = self.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        lines = [
            "FrameVitals custom checks",
            f"Status          {self.status.upper()}",
            f"Checks          {summary.get('checks', 0)}",
            f"Passed          {summary.get('passed', 0)}",
            f"Errors          {summary.get('errors', 0)}",
            f"Warnings        {summary.get('warnings', 0)}",
        ]
        if self.findings:
            lines.extend(["", "Findings"])
            for finding in self.findings[:10]:
                lines.append(
                    f"- [{str(finding.get('severity', 'error')).upper()}] "
                    f"{finding.get('title')}: {finding.get('message')}"
                )
        return "\n".join(lines)


class ValidationResult(_QualityResult):
    """Backward-compatible mapping returned by :func:`framevitals.validate`."""

    @property
    def valid(self) -> bool:
        return bool(self.get("valid", False))

    @property
    def status(self) -> str:
        return str(self.get("status", "unknown"))

    @property
    def findings(self) -> list[dict[str, Any]]:
        value = self.get("findings", [])
        return value if isinstance(value, list) else []

    def summary_text(self) -> str:
        summary = self.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        lines = [
            "FrameVitals validation",
            f"Status          {self.status.upper()}",
            f"Columns checked {summary.get('columns_checked', 0)}",
            f"Errors          {summary.get('errors', 0)}",
            f"Warnings        {summary.get('warnings', 0)}",
        ]
        if self.findings:
            lines.extend(["", "Top findings"])
            for finding in self.findings[:8]:
                lines.append(
                    f"- [{str(finding.get('severity', 'error')).upper()}] "
                    f"{finding.get('column')}: {finding.get('message')}"
                )
        return "\n".join(lines)


class DriftResult(_QualityResult):
    """Backward-compatible mapping returned by :func:`framevitals.compare`."""

    @property
    def severity(self) -> str:
        gate = self.get("gate", {})
        if isinstance(gate, dict):
            return str(gate.get("severity", "unknown"))
        return "unknown"

    @property
    def status(self) -> str:
        gate = self.get("gate", {})
        if isinstance(gate, dict):
            return str(gate.get("status", "unknown"))
        return "unknown"

    @property
    def columns(self) -> list[dict[str, Any]]:
        value = self.get("columns", [])
        return value if isinstance(value, list) else []

    def summary_text(self) -> str:
        if not self.get("available"):
            return (
                "FrameVitals drift\n"
                "Status          UNAVAILABLE\n"
                f"Reason          {self.get('reason')}"
            )

        summary = self.get("summary", {})
        schema = self.get("schema", {})
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(schema, dict):
            schema = {}
        lines = [
            "FrameVitals drift",
            f"Gate            {self.status.upper()}",
            f"Severity        {self.severity.upper()}",
            f"Columns checked {summary.get('n_columns_compared', 0)}",
            f"Added columns   {len(schema.get('added_columns', []))}",
            f"Removed columns {len(schema.get('removed_columns', []))}",
            f"Type changes    {len(schema.get('dtype_changes', []))}",
        ]
        notable = [
            entry
            for entry in self.columns
            if entry.get("drift_severity") in {"minor", "moderate", "severe"}
        ]
        if notable:
            lines.extend(["", "Top drift"])
            for entry in notable[:8]:
                lines.append(
                    f"- [{str(entry.get('drift_severity')).upper()}] {entry.get('column')}"
                )
        return "\n".join(lines)


class GateResult(_QualityResult):
    """Combined contract, custom-check, and drift quality-gate result."""

    @property
    def status(self) -> str:
        return str(self.get("status", "unknown"))

    @property
    def passed(self) -> bool:
        return bool(self.get("passed", False))

    @property
    def reasons(self) -> list[str]:
        value = self.get("reasons", [])
        return value if isinstance(value, list) else []

    @property
    def checks_run(self) -> list[str]:
        value = self.get("checks_run", [])
        return value if isinstance(value, list) else []

    def summary_text(self) -> str:
        checks = self.get("checks", {})
        if not isinstance(checks, dict):
            checks = {}
        validation = checks.get("validation")
        drift = checks.get("drift")
        custom = checks.get("custom")

        lines = [
            "FrameVitals quality gate",
            f"Status          {self.status.upper()}",
            f"Passed          {'YES' if self.passed else 'NO'}",
        ]
        if isinstance(validation, dict):
            lines.append(
                f"Validation      {str(validation.get('status', 'unknown')).upper()}"
            )
        if isinstance(drift, dict):
            drift_gate = drift.get("gate", {})
            if isinstance(drift_gate, dict):
                lines.append(
                    "Drift           "
                    f"{str(drift_gate.get('severity', 'unknown')).upper()}"
                )
        if isinstance(custom, dict):
            summary = custom.get("summary", {})
            if not isinstance(summary, dict):
                summary = {}
            lines.append(
                "Custom checks   "
                f"{str(custom.get('status', 'unknown')).upper()} "
                f"({summary.get('checks', 0)} run)"
            )
        if self.reasons:
            lines.extend(["", "Reasons"])
            for reason in self.reasons[:10]:
                lines.append(f"- {reason}")
        return "\n".join(lines)
