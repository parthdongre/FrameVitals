"""
Dataset brief packer.

Builds a compact, model-friendly JSON description of the analyzed dataset.
The brief is what the LLM should see *before* any specific question — it's
the single source of truth that lets the agent answer concretely without
hallucinating column names or numbers.

What goes in:
  - Identity:     id, filename, mode, rows × cols, target.
  - Schema:       per-column dtype, role tags, top categorical values.
  - Quality:      missing %, duplicates, health score components.
  - Analytics:    flagged signals, anomaly headline, leaderboard winner,
                  multicollinearity / leakage flags, time-series headline,
                  text-profile headline.
  - Sample:       first ~5 rows from `profile.preview`.
  - Narrative:    the AI report text if available (truncated).

The whole thing is JSON-serializable and capped at a configurable byte
budget so it fits inside any reasonable model context window.
"""
from __future__ import annotations

import json
import math
from typing import Any


__all__ = [
    "build_dataset_brief",
    "render_brief_block",
]

# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _safe_num(v: Any, default: float | int | None = None) -> float | int | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and isinstance(v, float):
        return v if math.isfinite(v) else default
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _round(v: Any, digits: int = 3) -> Any:
    n = _safe_num(v)
    if isinstance(n, (int, float)) and not isinstance(n, bool):
        return round(float(n), digits)
    return v


def _truncate_str(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def _take_dict(d: Any, max_keys: int) -> dict:
    if not isinstance(d, dict):
        return {}
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(d.items()):
        if i >= max_keys:
            break
        out[str(k)] = v
    return out


def _sorted_top(d: Any, max_keys: int, reverse: bool = True) -> dict:
    """Top-N items from a dict, sorted by numeric value."""
    if not isinstance(d, dict):
        return {}
    items = []
    for k, v in d.items():
        n = _safe_num(v)
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            items.append((str(k), float(n)))
    items.sort(key=lambda x: x[1], reverse=reverse)
    return {k: round(v, 4) for k, v in items[:max_keys]}


# ----------------------------------------------------------------------------
# Section builders
# ----------------------------------------------------------------------------


def _identity(p: dict) -> dict:
    profile = p.get("profile") or {}
    shape = profile.get("shape") or {}
    return {
        "id": p.get("id") or p.get("dataset_id"),
        "filename": p.get("filename"),
        "analysis_mode": p.get("analysisMode") or p.get("analysis_mode"),
        "rows": p.get("rows") or shape.get("rows"),
        "columns": p.get("columns") or shape.get("columns"),
        "target_column": p.get("selectedTargetColumn") or p.get("selected_target_column"),
        "target_candidates": (p.get("targetCandidates") or [])[:5],
        "data_types": p.get("dataTypes") or [],
    }


def _schema(p: dict, max_columns: int) -> dict:
    profile = p.get("profile") or {}
    dtypes = profile.get("dtypes") or {}
    missing_percent = profile.get("missing_percent") or {}
    column_roles = p.get("columnRoles") or p.get("column_roles") or {}
    cat_summary = profile.get("categorical_summary") or {}

    columns: list[dict] = []
    for i, (col, dtype) in enumerate(dtypes.items()):
        if i >= max_columns:
            break
        roles = column_roles.get(col)
        role_list: list[str] = []
        if isinstance(roles, dict):
            r = roles.get("roles")
            if isinstance(r, list):
                role_list = [str(x) for x in r][:4]
        # Top-3 categorical values + cardinality for low-cardinality cats.
        cat_info: dict[str, Any] = {}
        if isinstance(cat_summary, dict) and col in cat_summary:
            entry = cat_summary[col] or {}
            top_vals = entry.get("top_values") if isinstance(entry, dict) else None
            unique_count = entry.get("unique_count") if isinstance(entry, dict) else None
            if isinstance(unique_count, (int, float)) and not isinstance(unique_count, bool):
                cat_info["unique_count"] = int(unique_count)
            if isinstance(top_vals, dict):
                cat_info["top_values"] = _take_dict(top_vals, 3)
        columns.append({
            "name": col,
            "dtype": str(dtype),
            "missing_pct": _round(missing_percent.get(col), 2),
            "roles": role_list,
            **cat_info,
        })

    # Top columns by missing percent — useful for "which column has most missing values"
    missing_top = _sorted_top(missing_percent, max_keys=8, reverse=True)

    return {
        "numeric_columns": (profile.get("numeric_columns") or [])[:30],
        "categorical_columns": (profile.get("categorical_columns") or [])[:30],
        "date_columns": (profile.get("date_columns") or [])[:10],
        "duplicate_rows": profile.get("duplicate_rows"),
        "missing_top": missing_top,
        "columns": columns,
    }


def _quality(p: dict) -> dict:
    health = p.get("health") or {}
    ml = p.get("mlReadiness") or p.get("ml_readiness") or {}
    components = health.get("components") or {}
    if isinstance(components, dict):
        components = {k: _round(v, 2) for k, v in components.items()}
    return {
        "health_score": _round(health.get("overall_score"), 1),
        "health_label": health.get("label"),
        "components": components,
        "ml_readiness_score": _round(ml.get("score"), 1),
        "ml_readiness_label": ml.get("label"),
        "ml_recommendations": (ml.get("recommendations") or [])[:6],
    }


def _signals(p: dict, max_signals: int = 6) -> list[dict]:
    sigs = p.get("signals") or []
    out = []
    for s in sigs[:max_signals]:
        if not isinstance(s, dict):
            continue
        out.append({
            "name": s.get("name"),
            "severity": s.get("severity"),
            "evidence": _truncate_str(str(s.get("evidence") or ""), 200),
            "recommendation": _truncate_str(str(s.get("recommendation") or ""), 200),
        })
    return out


def _ml(p: dict) -> dict:
    out: dict[str, Any] = {}

    # Target analysis
    ta = p.get("targetAnalysis") or p.get("target_analysis") or {}
    if isinstance(ta, dict) and ta.get("available"):
        out["target_analysis"] = {
            "task_type": ta.get("task_type"),
            "n_classes": ta.get("n_classes"),
            "imbalance_ratio": _round(ta.get("imbalance_ratio"), 3),
            "missing_rate": _round(ta.get("missing_rate"), 4),
        }

    # Leaderboard winner
    lb = p.get("modelLeaderboard") or p.get("model_leaderboard") or {}
    if isinstance(lb, dict) and lb.get("available"):
        winner = lb.get("winner") or {}
        rows = lb.get("leaderboard") or []
        top3 = []
        for r in rows[:3]:
            if isinstance(r, dict):
                top3.append({
                    "model": r.get("model"),
                    "primary_score": _round(r.get("primary_score"), 4),
                })
        out["leaderboard"] = {
            "task_type": lb.get("task_type"),
            "primary_metric": lb.get("primary_metric"),
            "n_rows": lb.get("n_rows"),
            "n_features": lb.get("n_features"),
            "winner": winner.get("model") if isinstance(winner, dict) else None,
            "winner_score": _round(winner.get("primary_score") if isinstance(winner, dict) else None, 4),
            "top_3": top3,
        }

    # SHAP / explainability
    exp = p.get("explainability") or {}
    if isinstance(exp, dict) and exp.get("available"):
        gi = exp.get("global_importance") or []
        top_features = []
        for f in gi[:8]:
            if isinstance(f, dict):
                top_features.append({
                    "feature": f.get("feature"),
                    "importance": _round(f.get("importance"), 4),
                })
        out["explainability"] = {
            "model": exp.get("model"),
            "method": exp.get("method"),
            "top_features": top_features,
        }

    # Multicollinearity (high-VIF)
    mc = p.get("multicollinearity") or {}
    vif = (mc.get("vif") if isinstance(mc, dict) else None) or {}
    if isinstance(vif, dict) and vif.get("available"):
        scores = vif.get("vif_scores") or []
        top_vif = []
        for s in scores[:6]:
            if isinstance(s, dict):
                top_vif.append({
                    "feature": s.get("feature"),
                    "vif": _round(s.get("vif"), 2),
                    "severity": s.get("severity"),
                })
        out["multicollinearity"] = {
            "high_vif_count": vif.get("high_vif_count"),
            "medium_vif_count": vif.get("medium_vif_count"),
            "top_vif": top_vif,
        }

    # Target leakage
    tl = p.get("targetLeakage") or p.get("target_leakage") or {}
    if isinstance(tl, dict) and tl.get("available"):
        suspects = tl.get("suspect_features") or []
        out["target_leakage"] = {
            "status": tl.get("status"),
            "suspect_features": [
                {
                    "feature": s.get("feature") or s.get("column"),
                    "correlation": _round(s.get("correlation"), 3),
                    "mutual_information": _round(s.get("mutual_information"), 3),
                    "reason": _truncate_str(str(s.get("reason") or ""), 160),
                }
                for s in suspects[:6] if isinstance(s, dict)
            ],
        }

    # Diagnostics summary
md = (
    p.get("modelDiagnostics")
    or p.get("model_diagnostics")
    or {}
)

if isinstance(md, dict) and md.get(
    "available"
):
    diagnostics = {
        "task_type": md.get(
            "task_type"
        ),
        "target_column": md.get(
            "target_column"
        ),
    }

    if md.get("task_type") == (
        "classification"
    ):
        diagnostics.update({
            "accuracy": _round(
                md.get("accuracy"),
                4,
            ),
            "precision_weighted": _round(
                md.get(
                    "precision_weighted"
                ),
                4,
            ),
            "recall_weighted": _round(
                md.get(
                    "recall_weighted"
                ),
                4,
            ),
            "f1_weighted": _round(
                md.get("f1_weighted"),
                4,
            ),
        })

        cv = (
            md.get("cross_validation")
            or {}
        )

        diagnostics[
            "mean_cv_f1_weighted"
        ] = _round(
            cv.get(
                "mean_f1_weighted"
            ),
            4,
        )

    elif md.get("task_type") == (
        "regression"
    ):
        residuals = (
            md.get("residual_summary")
            or {}
        )

        diagnostics.update({
            "mean_abs_residual": _round(
                residuals.get(
                    "mean_abs_residual"
                ),
                4,
            ),
            "std_residual": _round(
                residuals.get(
                    "std_residual"
                ),
                4,
            ),
        })

        cv = (
            md.get("cross_validation")
            or {}
        )

        diagnostics[
            "mean_cv_r2"
        ] = _round(
            cv.get("mean_r2"),
            4,
        )

    out["model_diagnostics"] = {
        key: value
        for key, value
        in diagnostics.items()
        if value is not None
    }


def _time_text_drift(p: dict) -> dict:
    out: dict[str, Any] = {}

    ts = p.get("timeSeries") or p.get("time_series") or {}
    if isinstance(ts, dict) and ts.get("available"):
        out["time_series"] = {
            "date_column": ts.get("detected_date_column"),
            "frequency": ts.get("frequency"),
            "stationary": (ts.get("stationarity") or {}).get("is_stationary"),
            "period_estimate": ts.get("period_estimate"),
        }

    tp = p.get("textProfile") or p.get("text_profile") or {}
    if isinstance(tp, dict) and tp.get("available"):
        cols = tp.get("profiled_columns") or []
        out["text"] = {"profiled_columns": cols[:5]}

    return out


def _sample(p: dict, max_rows: int, max_cell_chars: int = 80) -> list[dict]:
    profile = p.get("profile") or {}
    rows = profile.get("preview") or []
    out: list[dict] = []
    for row in rows[:max_rows]:
        if not isinstance(row, dict):
            continue
        clipped: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str):
                clipped[k] = _truncate_str(v, max_cell_chars)
            elif isinstance(v, float) and not math.isfinite(v):
                clipped[k] = None
            else:
                clipped[k] = v
        out.append(clipped)
    return out


def _narrative(p: dict, max_chars: int) -> dict | None:
    ai = p.get("aiReport") or p.get("ai_report") or {}
    if not isinstance(ai, dict):
        return None
    text = ai.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "source": ai.get("source"),
        "text": _truncate_str(text, max_chars),
    }


def _correlations(p: dict, top_n: int = 6) -> list[dict]:
    """Return the top-N off-diagonal correlations by |value|."""
    profile = p.get("profile") or {}
    matrix = profile.get("correlations")
    if not isinstance(matrix, dict):
        return []
    pairs: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    for row, sub in matrix.items():
        if not isinstance(sub, dict):
            continue
        for col, v in sub.items():
            if row == col:
                continue
            key = tuple(sorted([str(row), str(col)]))
            if key in seen:
                continue
            n = _safe_num(v)
            if not isinstance(n, (int, float)) or isinstance(n, bool):
                continue
            seen.add(key)
            pairs.append((str(row), str(col), float(n)))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return [
        {"a": a, "b": b, "r": round(v, 3)}
        for a, b, v in pairs[:top_n]
    ]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def build_dataset_brief(
    analysis_result: dict | None,
    *,
    max_columns: int = 30,
    max_sample_rows: int = 5,
    max_narrative_chars: int = 1200,
    max_total_chars: int = 9000,
) -> dict:
    """Pack the analysis result into a compact dict the LLM can read.

    Pass `analysis_result` (the cached payload from /api/analyze).
    Returns a dict that's safe to JSON-serialize and (with default budgets)
    fits in roughly 2-3k tokens.

    The output is hard-capped at `max_total_chars` of serialized JSON; once
    that limit is hit, the lowest-priority sections (narrative, sample,
    correlations) are progressively dropped.
    """
    p = analysis_result or {}

    brief: dict[str, Any] = {
        "dataset": _identity(p),
        "schema": _schema(p, max_columns=max_columns),
        "quality": _quality(p),
        "signals": _signals(p),
        "ml": _ml(p),
        "time_text_drift": _time_text_drift(p),
        "correlations_top": _correlations(p, top_n=6),
        "sample_rows": _sample(p, max_rows=max_sample_rows),
        "ai_narrative": _narrative(p, max_chars=max_narrative_chars),
    }

    # Drop empty sections to keep the prompt tight.
    for k in list(brief.keys()):
        v = brief[k]
        if v in (None, [], {}):
            brief.pop(k)

    # Enforce the byte budget by progressively dropping the heaviest
    # optional sections. Order: narrative > sample_rows > correlations_top
    # > schema.columns trimmed to half.
    def _serialized() -> str:
        return json.dumps(brief, default=str, ensure_ascii=False)

    if len(_serialized()) > max_total_chars and "ai_narrative" in brief:
        brief["ai_narrative"]["text"] = _truncate_str(
            brief["ai_narrative"].get("text") or "", 600
        )
    if len(_serialized()) > max_total_chars and "sample_rows" in brief:
        brief["sample_rows"] = brief["sample_rows"][:3]
    if len(_serialized()) > max_total_chars and "correlations_top" in brief:
        brief["correlations_top"] = brief["correlations_top"][:3]
    if len(_serialized()) > max_total_chars and "ai_narrative" in brief:
        brief.pop("ai_narrative", None)
    if len(_serialized()) > max_total_chars and "sample_rows" in brief:
        brief.pop("sample_rows", None)

    if len(_serialized()) > max_total_chars:
        # Last resort: trim the per-column schema list.
        cols = (brief.get("schema") or {}).get("columns") or []
        brief.setdefault("schema", {})["columns"] = cols[: max_columns // 2]

    return brief


def render_brief_block(brief: dict, *, label: str = "Dataset brief") -> str:
    """Render the brief as a markdown-ish block for inclusion in a prompt."""
    body = json.dumps(brief, default=str, indent=2, ensure_ascii=False)
    return f"### {label} (JSON)\n```json\n{body}\n```"
