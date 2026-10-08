"""Self-contained HTML rendering for FrameVitals analysis results."""

from __future__ import annotations

import json
from html import escape
from typing import Any, Mapping


def _text(value: Any) -> str:
    return escape(str(value if value is not None else ""))


def _score(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _severity_class(value: Any) -> str:
    severity = str(value or "info").lower()
    if severity in {"critical", "high", "medium", "low", "info"}:
        return severity
    return "info"


def _source_kind(result: Mapping[str, Any]) -> str:
    return str(result.get("source_kind") or "tabular").lower()


def _source_metrics(result: Mapping[str, Any]) -> list[tuple[str, Any]]:
    kind = _source_kind(result)

    if kind == "graph":
        graph = result.get("graph", {}) or {}
        components = graph.get("components", {}) or {}
        spectral = graph.get("spectral", {}) or {}
        return [
            ("Nodes", graph.get("nodes", "n/a")),
            ("Edges", graph.get("edges", "n/a")),
            ("Density", graph.get("density", "n/a")),
            ("Components", components.get("component_count", "n/a")),
            ("Largest component", components.get("largest_component_ratio", "n/a")),
            ("Max core", (graph.get("core", {}) or {}).get("max_core", "n/a")),
            ("Algebraic connectivity", spectral.get("algebraic_connectivity", "n/a")),
        ]

    if kind == "tensor":
        tensor = result.get("tensor", {}) or {}
        matrix = tensor.get("matrix", {}) or {}
        return [
            ("Shape", tensor.get("shape", "n/a")),
            ("Dtype", tensor.get("dtype", "n/a")),
            ("Values", tensor.get("size", "n/a")),
            ("Zero fraction", tensor.get("zero_fraction_sample", "n/a")),
            ("Rank ratio", matrix.get("rank_ratio", "n/a")),
            ("Effective rank", matrix.get("effective_rank", "n/a")),
        ]

    if kind == "document":
        doc = result.get("document", {}) or {}
        return [
            ("Format", str(doc.get("format", "unknown")).upper()),
            ("Pages / slides", doc.get("pages", doc.get("slides", "n/a"))),
            ("Pages inspected", doc.get("pages_analyzed", "n/a")),
            ("Words", doc.get("words", 0)),
            ("Characters", doc.get("characters", 0)),
            ("Lines", doc.get("lines", 0)),
            ("Duplicate lines", doc.get("duplicate_lines", 0)),
            ("Inspection truncated", doc.get("truncated", False)),
        ]

    if kind == "model":
        model = result.get("model", {}) or {}
        runtime = model.get("runtime", {}) or {}
        runtime_summary = runtime.get("summary", {}) if isinstance(runtime, Mapping) else {}
        return [
            ("Framework", model.get("framework", "n/a")),
            ("Architecture", model.get("architecture", "n/a")),
            ("Parameters", model.get("parameters", "n/a")),
            ("Trainable", model.get("trainable_parameters", "n/a")),
            ("Modules", model.get("modules", model.get("nodes", "n/a"))),
            ("Runtime issues", sum(
                int(value)
                for value in runtime_summary.values()
                if isinstance(value, (int, float))
            ) if runtime_summary else "n/a"),
        ]

    if kind == "nested":
        nested = result.get("nested", {}) or {}
        return [
            ("Nodes observed", nested.get("nodes_observed", "n/a")),
            ("Max depth", nested.get("max_depth", "n/a")),
            ("Type conflicts", nested.get("path_type_conflict_count", "n/a")),
            ("Null values", nested.get("null_values", "n/a")),
            ("Cyclic references", nested.get("cyclic_references", "n/a")),
        ]

    if kind == "relational":
        relational = result.get("relational", {}) or {}
        return [
            ("Tables", relational.get("table_count", "n/a")),
            ("Relationships", relational.get("relationship_count", "n/a")),
            ("Isolated tables", len(relational.get("isolated_tables", []) or [])),
            ("Rows observed", relational.get("rows_observed", "n/a")),
        ]

    profile = result.get("profile", {}) or {}
    shape = profile.get("shape", {}) or {}
    health = result.get("health", {}) or {}
    return [
        ("Rows", shape.get("rows", "n/a")),
        ("Columns", shape.get("columns", "n/a")),
        ("Missing cells", f"{health.get('details', {}).get('missing_percent', 0)}%"),
        ("Duplicate rows", f"{profile.get('duplicate_percent', 0)}%"),
        ("Memory", f"{profile.get('memory_usage_mb', 'n/a')} MB"),
    ]


def _structure_table(result: Mapping[str, Any]) -> str:
    kind = _source_kind(result)
    if kind == "tabular":
        return (
            '<div style="overflow:auto"><table><thead><tr>'
            '<th>Column</th><th>dtype</th><th>Missing</th><th>Unique</th><th>Roles</th>'
            f'</tr></thead><tbody>{_column_rows(result)}</tbody></table></div>'
        )

    rows = "".join(
        "<tr>"
        f"<th>{_text(label)}</th>"
        f"<td>{_text(value)}</td>"
        "</tr>"
        for label, value in _source_metrics(result)
    )
    return f'<div style="overflow:auto"><table><tbody>{rows}</tbody></table></div>'


def _column_rows(result: Mapping[str, Any]) -> str:
    profile = result.get("profile", {}) or {}
    roles = result.get("column_roles", {}) or {}
    columns = profile.get("columns", []) or []
    dtypes = profile.get("dtypes", {}) or {}
    missing = profile.get("missing_percent", {}) or {}

    rows: list[str] = []
    for name in columns:
        role_info = roles.get(name, {}) or {}
        role_text = ", ".join(role_info.get("roles", [])[:6])
        unique_count = role_info.get("unique_count", "")
        rows.append(
            "<tr>"
            f"<td><code>{_text(name)}</code></td>"
            f"<td>{_text(dtypes.get(name, ''))}</td>"
            f"<td>{_text(missing.get(name, ''))}%</td>"
            f"<td>{_text(unique_count)}</td>"
            f"<td>{_text(role_text)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _finding_cards(result: Mapping[str, Any]) -> str:
    findings = result.get("findings", []) or []
    if not findings:
        return '<div class="empty">No actionable findings from the current signal layer.</div>'

    cards: list[str] = []
    for finding in findings:
        severity = _severity_class(finding.get("severity"))
        recommendation = finding.get("recommendation") or ""
        recommendation_html = (
            f'<div class="recommendation"><strong>Next action</strong><br>{_text(recommendation)}</div>'
            if recommendation
            else ""
        )
        cards.append(
            f'<article class="finding {severity}">'
            f'<div class="finding-top"><span class="badge">{_text(severity.upper())}</span>'
            f'<span class="finding-code">{_text(finding.get("code", ""))}</span></div>'
            f'<h3>{_text(finding.get("title", "Finding"))}</h3>'
            f'<p>{_text(finding.get("evidence", ""))}</p>'
            f'{recommendation_html}'
            '</article>'
        )
    return "".join(cards)


def render_notebook_summary(result: Mapping[str, Any]) -> str:
    """Return a compact notebook-safe HTML representation."""
    health = result.get("health", {}) or {}
    findings = result.get("findings", []) or []
    kind = _source_kind(result)
    metrics = _source_metrics(result)
    metric_text = " · ".join(
        f"{_text(label)}: {_text(value)}"
        for label, value in metrics[:3]
    )

    return f"""
<div style="font-family:Inter,system-ui,-apple-system,sans-serif;border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;max-width:760px;background:#fff;color:#111827">
  <div style="display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:12px">
    <div><strong style="font-size:18px">FrameVitals</strong><div style="color:#6b7280;font-size:13px">{_text(result.get('filename', '<source>'))}</div></div>
    <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em">{_text(kind)}</div>
  </div>
  <div style="color:#475467;font-size:13px;margin-bottom:12px">{metric_text}</div>
  <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px">
    <div style="background:#f9fafb;padding:12px;border-radius:10px"><div style="font-size:12px;color:#6b7280">Health</div><strong>{_text(health.get('overall_score','n/a'))}/100</strong><div style="font-size:12px">{_text(health.get('label',''))}</div></div>
    <div style="background:#f9fafb;padding:12px;border-radius:10px"><div style="font-size:12px;color:#6b7280">Source</div><strong>{_text(kind.upper())}</strong><div style="font-size:12px">{_text(result.get('analysis_mode','standard'))}</div></div>
    <div style="background:#f9fafb;padding:12px;border-radius:10px"><div style="font-size:12px;color:#6b7280">Beacons</div><strong>{len(findings)}</strong><div style="font-size:12px">surfaced</div></div>
  </div>
</div>
""".strip()


def render_html_report(result: Mapping[str, Any]) -> str:
    """Render a complete, self-contained HTML analysis report."""
    profile = result.get("profile", {}) or {}
    shape = profile.get("shape", {}) or {}
    health = result.get("health", {}) or {}
    ml = result.get("ml_readiness", {}) or {}
    findings = result.get("findings", []) or []
    source_kind = _source_kind(result)
    source_metrics = _source_metrics(result)
    recommendations = []
    seen: set[str] = set()
    for finding in findings:
        recommendation = str(finding.get("recommendation") or "").strip()
        if recommendation and recommendation not in seen:
            seen.add(recommendation)
            recommendations.append(recommendation)

    health_score = _score(health.get("overall_score"))
    ml_score = _score(ml.get("score"))
    total_ms = (result.get("timings_ms", {}) or {}).get("total")
    duration = f"{float(total_ms) / 1000:.2f}s" if isinstance(total_ms, (int, float)) else "n/a"

    recommendation_items = "".join(
        f"<li>{_text(item)}</li>" for item in recommendations
    ) or "<li>No additional remediation steps are required by the current finding layer.</li>"

    if source_kind == "tabular":
        hero_meta = "".join([
            f"<span>{_text(shape.get('rows','?'))} rows</span>",
            f"<span>{_text(shape.get('columns','?'))} columns</span>",
        ])
        source_card = (
            f'<article class="panel score"><div class="label">ML readiness</div>'
            f'<div class="big">{_text(ml.get("score","n/a"))}'
            '<span style="font-size:17px;color:#98a2b3"> / 100</span></div>'
            f'<div>{_text(ml.get("label",""))}</div>'
            f'<div class="bar"><span style="width:{ml_score:.1f}%"></span></div></article>'
        )
    else:
        hero_meta = "".join(
            f"<span>{_text(label)}: {_text(value)}</span>"
            for label, value in source_metrics[:3]
        )
        source_card = (
            '<article class="panel score"><div class="label">Source kind</div>'
            f'<div class="big" style="font-size:30px">{_text(source_kind.upper())}</div>'
            '<div style="color:#667085">Structure-aware Prism diagnostics</div></article>'
        )

    metric_cards = "".join(
        '<article class="panel metric">'
        f'<div class="label">{_text(label)}</div>'
        f'<div class="big" style="font-size:24px">{_text(value)}</div>'
        '</article>'
        for label, value in source_metrics[:4]
    )

    raw_json = escape(json.dumps(dict(result), indent=2, default=str))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FrameVitals Report — {_text(result.get('filename', 'dataset'))}</title>
<style>
:root {{ color-scheme: light; --bg:#f5f7fb; --panel:#ffffff; --ink:#111827; --muted:#687386; --line:#e6eaf0; --accent:#3157d5; --good:#15803d; --warn:#b45309; --bad:#b42318; }}
* {{ box-sizing:border-box; }}
body {{ margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.5; }}
.wrap {{ max-width:1180px;margin:0 auto;padding:42px 22px 70px; }}
.hero {{ background:linear-gradient(135deg,#111827 0%,#1f2a44 62%,#3157d5 100%);color:white;border-radius:24px;padding:34px;box-shadow:0 18px 60px rgba(17,24,39,.16); }}
.eyebrow {{ font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;opacity:.72; }}
h1 {{ margin:7px 0 6px;font-size:clamp(32px,5vw,58px);line-height:1.04; }}
.hero p {{ margin:0;color:#d7ddec; }}
.meta {{ display:flex;gap:10px;flex-wrap:wrap;margin-top:22px; }}
.meta span {{ padding:7px 10px;border:1px solid rgba(255,255,255,.18);border-radius:999px;background:rgba(255,255,255,.07);font-size:13px; }}
.grid {{ display:grid;grid-template-columns:repeat(12,1fr);gap:16px;margin-top:18px; }}
.panel {{ background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:0 5px 18px rgba(17,24,39,.035); }}
.score {{ grid-column:span 4;min-height:170px; }}
.metric {{ grid-column:span 3; }}
.section {{ grid-column:1/-1; }}
.label {{ color:var(--muted);font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase; }}
.big {{ font-size:36px;font-weight:800;margin-top:7px; }}
.bar {{ height:9px;background:#edf0f5;border-radius:999px;overflow:hidden;margin:18px 0 9px; }}
.bar > span {{ display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#3157d5,#6d8aff); }}
.findings {{ display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px; }}
.finding {{ border:1px solid var(--line);border-left-width:4px;border-radius:14px;padding:16px;background:#fff; }}
.finding.critical,.finding.high {{ border-left-color:#d92d20; }}
.finding.medium {{ border-left-color:#f79009; }}
.finding.low {{ border-left-color:#2e90fa; }}
.finding.info {{ border-left-color:#667085; }}
.finding-top {{ display:flex;justify-content:space-between;gap:10px;align-items:center; }}
.badge {{ font-size:10px;font-weight:900;letter-spacing:.08em;padding:4px 7px;border-radius:999px;background:#f2f4f7; }}
.finding-code {{ color:var(--muted);font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.finding h3 {{ margin:10px 0 5px;font-size:17px; }}
.finding p {{ margin:0;color:#475467;font-size:14px; }}
.recommendation {{ margin-top:12px;padding:10px 12px;background:#f8fafc;border-radius:10px;font-size:13px;color:#344054; }}
table {{ width:100%;border-collapse:collapse;margin-top:12px;font-size:13px; }}
th,td {{ text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top; }}
th {{ color:#475467;font-size:11px;text-transform:uppercase;letter-spacing:.05em; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.95em; }}
ul {{ padding-left:20px; }}
details {{ margin-top:12px; }}
summary {{ cursor:pointer;font-weight:700; }}
pre {{ overflow:auto;background:#111827;color:#d1d5db;padding:16px;border-radius:12px;font-size:11px;max-height:520px; }}
.empty {{ color:var(--muted);padding:18px 0; }}
.footer {{ color:var(--muted);text-align:center;font-size:12px;margin-top:26px; }}
@media (max-width:820px) {{ .score,.metric {{ grid-column:1/-1; }} .findings {{ grid-template-columns:1fr; }} .hero {{ padding:25px; }} }}
</style>
</head>
<body>
<main class="wrap">
<section class="hero">
  <div class="eyebrow">FrameVitals analysis report</div>
  <h1>{_text(result.get('filename', '<dataframe>'))}</h1>
  <p>Structure-aware health, diagnostics, change signals, and actionable Beacons in one report.</p>
  <div class="meta">
    {hero_meta}
    <span>{_text(result.get('analysis_mode','unknown'))} depth</span>
    <span>{len(findings)} findings</span>
    <span>{_text(duration)} runtime</span>
  </div>
</section>

<section class="grid">
  <article class="panel score"><div class="label">Data health</div><div class="big">{_text(health.get('overall_score','n/a'))}<span style="font-size:17px;color:#98a2b3"> / 100</span></div><div>{_text(health.get('label',''))}</div><div class="bar"><span style="width:{health_score:.1f}%"></span></div></article>
  {source_card}
  <article class="panel score"><div class="label">Actionable findings</div><div class="big">{len(findings)}</div><div style="color:#667085">Normalized from FrameVitals' deterministic signal layer.</div></article>

  {metric_cards}

  <section class="panel section"><div class="label">Findings</div><h2>What needs attention</h2><div class="findings">{_finding_cards(result)}</div></section>

  <section class="panel section"><div class="label">Recommendations</div><h2>Suggested next actions</h2><ul>{recommendation_items}</ul></section>

  <section class="panel section"><div class="label">Structure</div><h2>{_text(source_kind.title())} overview</h2>{_structure_table(result)}</section>

  <section class="panel section"><div class="label">Complete result</div><details><summary>Inspect raw JSON</summary><pre>{raw_json}</pre></details></section>
</section>
<div class="footer">Generated locally by FrameVitals. This HTML file is self-contained and does not require a server.</div>
</main>
</body>
</html>
"""
