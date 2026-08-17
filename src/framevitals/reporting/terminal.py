"""Dependency-light terminal rendering for FrameVitals results."""

from __future__ import annotations

from typing import Any, Mapping


def _score_bar(value: Any, width: int = 24) -> str:
    try:
        score = max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return "[" + "?" * width + "]"
    filled = round(score / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.1f}/100"
    except (TypeError, ValueError):
        return "n/a"


def _clean_line(value: Any, *, max_length: int = 110) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def render_terminal_summary(result: Mapping[str, Any]) -> str:
    """Render a compact report suitable for interactive terminal output."""
    profile = result.get("profile", {}) or {}
    shape = profile.get("shape", {}) or {}
    health = result.get("health", {}) or {}
    ml = result.get("ml_readiness", {}) or {}
    findings = result.get("findings", []) or []
    timings = result.get("timings_ms", {}) or {}

    rows = shape.get("rows", "?")
    columns = shape.get("columns", "?")
    health_score = health.get("overall_score")
    ml_score = ml.get("score")

    lines = [
        "FrameVitals Analysis",
        "=" * 72,
        f"Dataset       {result.get('filename', '<unknown>')}",
        f"Mode          {result.get('analysis_mode', 'unknown')}",
        f"Shape         {rows} rows x {columns} columns",
        f"Memory        {profile.get('memory_usage_mb', 'n/a')} MB",
        "",
        f"Health        {_score_bar(health_score)}  {_fmt_score(health_score)}  {health.get('label', '')}",
        f"ML readiness  {_score_bar(ml_score)}  {_fmt_score(ml_score)}  {ml.get('label', '')}",
        "",
        f"Findings      {len(findings)} actionable issue(s)",
    ]

    if findings:
        for finding in findings[:6]:
            severity = str(finding.get("severity", "info")).upper()
            title = _clean_line(finding.get("title", "Finding"), max_length=42)
            evidence = _clean_line(finding.get("evidence", ""), max_length=88)
            lines.append(f"  [{severity:<8}] {title}")
            if evidence:
                lines.append(f"             {evidence}")
        if len(findings) > 6:
            lines.append(f"  ... and {len(findings) - 6} more finding(s)")
    else:
        lines.append("  No actionable findings from the current signal layer.")

    total_ms = timings.get("total")
    if isinstance(total_ms, (int, float)):
        lines.extend(["", f"Completed in   {total_ms / 1000:.2f}s"])

    lines.extend([
        "=" * 72,
        "Use result.to_html(...) or CLI --output to keep the complete report.",
    ])
    return "\n".join(lines)
