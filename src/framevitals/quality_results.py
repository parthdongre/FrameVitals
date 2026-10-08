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
                "FrameVitals Tide\n"
                "Status          UNAVAILABLE\n"
                f"Reason          {self.get('reason')}"
            )

        summary = self.get("summary", {})
        schema = self.get("schema", {})
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(schema, dict):
            schema = {}

        source_kind = str(self.get("source_kind") or "tabular")
        if source_kind == "tabular":
            lines = [
                "FrameVitals drift",
                f"Gate            {self.status.upper()}",
                f"Severity        {self.severity.upper()}",
            ]
        else:
            lines = [
                "FrameVitals · Tide",
                f"Kind            {source_kind.upper()}",
                f"Gate            {self.status.upper()}",
                f"Severity        {self.severity.upper()}",
            ]

        if source_kind == "document":
            document = self.get("document", {})
            if not isinstance(document, dict):
                document = {}
            reference = document.get("reference", {})
            current = document.get("current", {})
            lines.extend([
                f"Format          {reference.get('format', '?')} -> {current.get('format', '?')}",
                f"Words           {reference.get('words', '?')} -> {current.get('words', '?')}",
                f"Lines           {reference.get('lines', '?')} -> {current.get('lines', '?')}",
                f"Text changed    {document.get('text_changed', False)}",
            ])
            return "\n".join(lines)

        if source_kind == "model":
            model = self.get("model", {})
            if not isinstance(model, dict):
                model = {}
            lines.extend([
                f"Compared        {model.get('parameters_compared', 0)} parameters",
                f"Added           {len(model.get('added_parameters', []))}",
                f"Removed         {len(model.get('removed_parameters', []))}",
                f"Shape changes   {len(model.get('shape_changes', []))}",
                f"P95 movement    {model.get('p95_relative_l2', 'n/a')}",
            ])
            most_changed = model.get("most_changed", [])
            if isinstance(most_changed, list) and most_changed:
                lines.extend(["", "Most changed"])
                for entry in most_changed[:8]:
                    lines.append(
                        f"- {entry.get('name')}: relative L2={entry.get('relative_l2')}"
                    )
            return "\n".join(lines)

        if source_kind == "nested":
            nested = self.get("nested", {})
            if not isinstance(nested, dict):
                nested = {}
            lines.extend([
                f"Nodes           {nested.get('reference_nodes', '?')} -> {nested.get('current_nodes', '?')}",
                f"Depth           {nested.get('reference_depth', '?')} -> {nested.get('current_depth', '?')}",
                f"Added keys      {len(nested.get('added_keys', []))}",
                f"Removed keys    {len(nested.get('removed_keys', []))}",
                f"Type conflicts  {nested.get('reference_type_conflicts', '?')} -> {nested.get('current_type_conflicts', '?')}",
            ])
            return "\n".join(lines)

        if source_kind == "relational":
            relational = self.get("relational", {})
            if not isinstance(relational, dict):
                relational = {}
            lines.extend([
                f"Tables          {relational.get('reference_table_count', '?')} -> {relational.get('current_table_count', '?')}",
                f"Added tables    {len(relational.get('added_tables', []))}",
                f"Removed tables  {len(relational.get('removed_tables', []))}",
                f"Row changes     {len(relational.get('row_changes', {}))}",
                f"Schema changes  {len(relational.get('column_changes', {}))}",
            ])
            return "\n".join(lines)

        if source_kind == "graph":
            graph = self.get("graph", {})
            if not isinstance(graph, dict):
                graph = {}
            lines.extend([
                f"Nodes           {graph.get('reference_nodes', '?')} -> {graph.get('current_nodes', '?')}",
                f"Edges           {graph.get('reference_edges', '?')} -> {graph.get('current_edges', '?')}",
                f"Node overlap    {graph.get('node_jaccard', 'n/a')}",
                f"Degree change   {graph.get('degree_js_distance', 'n/a')}",
            ])
            return "\n".join(lines)

        if source_kind == "tensor":
            tensor = self.get("tensor", {})
            if not isinstance(tensor, dict):
                tensor = {}
            aligned = tensor.get("aligned", {})
            if not isinstance(aligned, dict):
                aligned = {}
            lines.extend([
                f"Shape           {tensor.get('reference_shape', '?')} -> {tensor.get('current_shape', '?')}",
                f"Dtype           {tensor.get('reference_dtype', '?')} -> {tensor.get('current_dtype', '?')}",
                f"Relative L2     {aligned.get('relative_l2', 'n/a')}",
                f"Cosine          {aligned.get('cosine_similarity', 'n/a')}",
                f"Wasserstein     {aligned.get('wasserstein', 'n/a')}",
            ])
            return "\n".join(lines)

        lines.extend([
            f"Columns checked {summary.get('n_columns_compared', 0)}",
            f"Added columns   {len(schema.get('added_columns', []))}",
            f"Removed columns {len(schema.get('removed_columns', []))}",
            f"Type changes    {len(schema.get('dtype_changes', []))}",
        ])
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
