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


def _source_lines(result: Mapping[str, Any]) -> list[str]:
    kind = str(result.get("source_kind") or "tabular").lower()
    if kind == "graph":
        graph = result.get("graph", {}) or {}
        components = graph.get("components", {}) or {}
        core = graph.get("core", {}) or {}
        spectral = graph.get("spectral", {}) or {}
        return [
            "Kind          GRAPH",
            f"Nodes         {graph.get('nodes', '?')}",
            f"Edges         {graph.get('edges', '?')}",
            f"Components    {components.get('component_count', '?')}",
            f"Max core      {core.get('max_core', 'n/a')}",
            f"Spectral λ₂   {spectral.get('algebraic_connectivity', 'n/a')}",
        ]
    if kind == "tensor":
        tensor = result.get("tensor", {}) or {}
        matrix = tensor.get("matrix", {}) or {}
        return [
            "Kind          TENSOR",
            f"Shape         {tensor.get('shape', '?')}",
            f"Dtype         {tensor.get('dtype', '?')}",
            f"Values        {tensor.get('size', '?')}",
            f"Rank ratio    {matrix.get('rank_ratio', 'n/a')}",
            f"Eff. rank     {matrix.get('effective_rank', 'n/a')}",
        ]
    if kind == "document":
        document = result.get("document", {}) or {}
        return [
            "Kind          DOCUMENT",
            f"Format        {document.get('format', 'unknown')}",
            f"Pages         {document.get('pages', document.get('slides', 'n/a'))}",
            f"Words         {document.get('words', 0)}",
            f"Characters    {document.get('characters', 0)}",
            f"Text missing  {document.get('empty_text', False)}",
            f"Partial scan  {document.get('truncated', False)}",
        ]
    if kind == "model":
        model = result.get("model", {}) or {}
        return [
            "Kind          MODEL",
            f"Framework     {model.get('framework', 'unknown')}",
            f"Architecture  {model.get('architecture', 'unknown')}",
            f"Parameters    {model.get('parameters', '?')}",
            f"Trainable     {model.get('trainable_parameters', 'n/a')}",
            f"Modules       {model.get('modules', model.get('nodes', 'n/a'))}",
        ]
    if kind == "nested":
        nested = result.get("nested", {}) or {}
        return [
            "Kind          NESTED",
            f"Nodes         {nested.get('nodes_observed', '?')}",
            f"Depth         {nested.get('max_depth', '?')}",
            f"Conflicts     {nested.get('path_type_conflict_count', '?')}",
            f"Cycles        {nested.get('cyclic_references', '?')}",
        ]
    if kind == "relational":
        relational = result.get("relational", {}) or {}
        return [
            "Kind          RELATIONAL",
            f"Tables        {relational.get('table_count', '?')}",
            f"Relationships {relational.get('relationship_count', '?')}",
            f"Isolated      {len(relational.get('isolated_tables', []) or [])}",
        ]

    profile = result.get("profile", {}) or {}
    shape = profile.get("shape", {}) or {}
    return [
        "Kind          TABULAR",
        f"Shape         {shape.get('rows', '?')} rows x {shape.get('columns', '?')} columns",
        f"Memory        {profile.get('memory_usage_mb', 'n/a')} MB",
    ]


def render_terminal_summary(result: Mapping[str, Any]) -> str:
    """Render a compact report suitable for interactive terminal output."""
    health = result.get("health", {}) or {}
    ml = result.get("ml_readiness", {}) or {}
    findings = result.get("findings", []) or []
    timings = result.get("timings_ms", {}) or {}

    health_score = health.get("overall_score")
    ml_score = ml.get("score")
    source_kind = str(result.get("source_kind") or "tabular").lower()

    lines = [
        "FrameVitals Analysis",
        "=" * 72,
        f"Source        {result.get('filename', '<unknown>')}",
        f"Depth         {result.get('analysis_mode', 'unknown')}",
        *_source_lines(result),
        "",
        f"Health        {_score_bar(health_score)}  {_fmt_score(health_score)}  {health.get('label', '')}",
    ]
    if source_kind == "tabular":
        lines.append(
            f"ML readiness  {_score_bar(ml_score)}  {_fmt_score(ml_score)}  {ml.get('label', '')}"
        )
    lines.extend([
        "",
        f"Beacons       {len(findings)} surfaced",
    ])

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
