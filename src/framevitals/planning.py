"""Analysis planning result objects and human-readable explanations."""

from __future__ import annotations

from typing import Any


class AnalysisPlan(dict):
    """Dict-compatible preview of analyses FrameVitals considers applicable."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def selection(self) -> dict[str, Any]:
        value = self.get("selection", {})
        return value if isinstance(value, dict) else {}

    @property
    def selected(self) -> list[dict[str, Any]]:
        value = self.selection.get("selected_analyses", [])
        return value if isinstance(value, list) else []

    @property
    def skipped(self) -> list[dict[str, Any]]:
        value = self.selection.get("skipped_analyses", [])
        return value if isinstance(value, list) else []

    @property
    def recommended(self) -> list[dict[str, Any]]:
        value = self.selection.get("recommended_analyses", [])
        return value if isinstance(value, list) else []

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_name": self.get("dataset_name"),
            "analysis_mode": self.get("analysis_mode"),
            "target": self.get("target"),
            "shape": self.get("shape", {}),
            "selected_count": len(self.selected),
            "skipped_count": len(self.skipped),
            "recommended_count": len(self.recommended),
        }

    def explain_text(self) -> str:
        """Render a terminal-friendly explanation of the current plan."""
        shape = self.get("shape", {}) or {}
        lines = [
            "FrameVitals Analysis Plan",
            "=" * 72,
            f"Dataset       {self.get('dataset_name', '<unknown>')}",
            f"Mode          {self.get('analysis_mode', 'unknown')}",
            f"Target        {self.get('target') or '<none>'}",
            f"Shape         {shape.get('rows', '?')} rows x {shape.get('columns', '?')} columns",
            "",
            f"Selected      {len(self.selected)}",
        ]

        for item in self.selected:
            lines.append(
                f"  [RUN]  {item.get('id', ''):<28} {item.get('name', '')}"
            )

        lines.extend(["", f"Recommended   {len(self.recommended)}"])
        for item in self.recommended:
            lines.append(
                f"  [NEXT] {item.get('id', ''):<28} {item.get('reason', '')}"
            )

        lines.extend(["", f"Skipped       {len(self.skipped)}"])
        for item in self.skipped[:12]:
            reason = " ".join(str(item.get("reason", "")).split())
            if len(reason) > 68:
                reason = reason[:67].rstrip() + "…"
            lines.append(
                f"  [SKIP] {item.get('id', ''):<28} {reason}"
            )
        if len(self.skipped) > 12:
            lines.append(f"  ... and {len(self.skipped) - 12} more skipped analyses")

        lines.extend([
            "=" * 72,
            "This is a preview only; no heavy model/statistics stage was executed.",
        ])
        return "\n".join(lines)
