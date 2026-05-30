"""
Agent Tool Registry (WS-9)
==========================
A small, audited set of tools the LLM-driven agent can call.

Each tool:
    - Takes JSON-serializable arguments
    - Returns a JSON-serializable dict
    - Never mutates the analysis result or dataframe
    - Has a stable name + description used in the planner prompt

Available tools:
    - get_section(path)             → Read a slice of the analysis result
    - list_columns()                → Return column names + dtypes
    - column_summary(name)          → Detailed summary for one column
    - run_query(expression)         → Run a sandboxed pandas expression
    - get_top_anomalies(k)          → Top-k anomalous rows from anomaly ensemble
    - get_leaderboard()             → Model leaderboard summary
    - get_explainability_top(k)     → Top-k SHAP global features
    - search_facts(question, k)     → Retrieve top-k RAG facts

The registry exposes:
    AGENT_TOOLS: dict[str, dict]   # name -> {description, parameters_schema, callable}
    run_tool(name, args, ctx)      # one-shot dispatcher with argument validation
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from modules.rag_index import retrieve as rag_retrieve
from modules.rag_index import render_facts_block
from modules.safe_pandas import safe_eval


# ---------------------------------------------------------------------------
# Context object passed to every tool
# ---------------------------------------------------------------------------

class AgentContext:
    """Bundles the runtime state every tool needs."""

    def __init__(
        self,
        df: pd.DataFrame | None,
        analysis_result: dict,
        facts: list | None = None,
    ) -> None:
        self.df = df
        self.result = analysis_result or {}
        self.facts = facts or []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walk_path(obj: Any, path: str) -> Any:
    """Resolve a dotted path like 'health.components.completeness' into the value."""
    if not path:
        return obj
    cursor = obj
    for part in path.split("."):
        if isinstance(cursor, dict):
            cursor = cursor.get(part)
        elif isinstance(cursor, list):
            try:
                idx = int(part)
                cursor = cursor[idx] if 0 <= idx < len(cursor) else None
            except ValueError:
                return None
        else:
            return None
        if cursor is None:
            return None
    return cursor


def _truncate(value: Any, max_items: int = 50, max_chars: int = 6000) -> Any:
    """Truncate large lists/dicts so a tool result fits in the prompt budget."""
    import json

    try:
        serialized = json.dumps(value, default=str)
    except Exception:
        return str(value)[:max_chars]

    if len(serialized) <= max_chars:
        return value

    if isinstance(value, list):
        return value[:max_items] + [{"__truncated__": True, "remaining": len(value) - max_items}]

    if isinstance(value, dict):
        out = {}
        used = 0
        for k, v in value.items():
            entry = json.dumps({k: v}, default=str)
            if used + len(entry) > max_chars:
                out["__truncated__"] = True
                break
            out[k] = v
            used += len(entry)
        return out

    return str(value)[:max_chars]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_get_section(args: dict, ctx: AgentContext) -> dict:
    path = (args.get("path") or "").strip()
    if not path:
        return {"ok": False, "error": "Missing 'path' argument."}
    value = _walk_path(ctx.result, path)
    if value is None:
        return {"ok": False, "error": f"Path not found: {path}"}
    return {"ok": True, "path": path, "value": _truncate(value)}


def _tool_list_columns(args: dict, ctx: AgentContext) -> dict:
    profile = ctx.result.get("profile", {})
    columns = profile.get("columns", [])
    dtypes = profile.get("dtypes", {})
    return {
        "ok": True,
        "columns": [{"name": c, "dtype": str(dtypes.get(c, "?"))} for c in columns],
    }


def _tool_column_summary(args: dict, ctx: AgentContext) -> dict:
    name = (args.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "Missing 'name' argument."}

    profile = ctx.result.get("profile", {})
    if name not in profile.get("columns", []):
        return {"ok": False, "error": f"Column not found: {name}"}

    summary: dict[str, Any] = {
        "name": name,
        "dtype": profile.get("dtypes", {}).get(name),
        "missing_count": profile.get("missing_counts", {}).get(name),
        "missing_percent": profile.get("missing_percent", {}).get(name),
    }

    numeric_summary = profile.get("numeric_summary", {}).get(name)
    if numeric_summary:
        summary["numeric_summary"] = numeric_summary

    cat_summary = profile.get("categorical_summary", {}).get(name)
    if cat_summary:
        summary["categorical_summary"] = cat_summary

    deep = ctx.result.get("deep_statistics_v2") or {}
    deep_numeric = deep.get("numeric_statistics", {}).get(name)
    if deep_numeric:
        summary["deep_statistics"] = deep_numeric

    column_roles = ctx.result.get("column_roles", {}).get(name)
    if column_roles:
        summary["roles"] = column_roles

    return {"ok": True, "summary": summary}


def _tool_run_query(args: dict, ctx: AgentContext) -> dict:
    if ctx.df is None:
        return {"ok": False, "error": "DataFrame not available in this context."}
    expr = (args.get("expression") or "").strip()
    if not expr:
        return {"ok": False, "error": "Missing 'expression' argument."}
    out = safe_eval(expr, ctx.df)
    out["truncated_preview"] = True
    return out


def _tool_get_top_anomalies(args: dict, ctx: AgentContext) -> dict:
    k = int(args.get("k", 10))
    anomalies = ctx.result.get("anomalies_v2") or {}
    if not anomalies.get("available"):
        return {"ok": False, "error": "Anomaly ensemble unavailable."}
    rows = (anomalies.get("top_rows") or [])[: max(1, min(k, 50))]
    return {
        "ok": True,
        "detectors": anomalies.get("detectors_run", []),
        "threshold": anomalies.get("threshold"),
        "flagged_count": anomalies.get("flagged_count"),
        "ensemble_summary": anomalies.get("ensemble_summary"),
        "top_rows": rows,
    }


def _tool_get_leaderboard(args: dict, ctx: AgentContext) -> dict:
    lb = ctx.result.get("model_leaderboard") or {}
    if not lb.get("available"):
        return {"ok": False, "error": lb.get("message") or "Leaderboard unavailable."}
    return {
        "ok": True,
        "task_type": lb.get("task_type"),
        "primary_metric": lb.get("primary_metric"),
        "winner": lb.get("winner"),
        "rows": (lb.get("leaderboard") or [])[:8],
    }


def _tool_get_explainability_top(args: dict, ctx: AgentContext) -> dict:
    k = int(args.get("k", 10))
    ex = ctx.result.get("explainability") or {}
    if not ex.get("available"):
        return {"ok": False, "error": ex.get("message") or "Explainability unavailable."}
    return {
        "ok": True,
        "method": ex.get("method"),
        "model": ex.get("model"),
        "global_importance": (ex.get("global_importance") or [])[: max(1, min(k, 20))],
        "per_row_stories": (ex.get("per_row_stories") or [])[:3],
    }


def _tool_search_facts(args: dict, ctx: AgentContext) -> dict:
    question = (args.get("question") or "").strip()
    k = int(args.get("k", 8))
    if not question:
        return {"ok": False, "error": "Missing 'question' argument."}
    if not ctx.facts:
        return {"ok": False, "error": "Fact index not built."}
    retrieved = rag_retrieve(question, ctx.facts, k=k)
    return {
        "ok": True,
        "backend": retrieved["backend"],
        "facts": retrieved["facts"],
        "rendered": render_facts_block(retrieved),
    }


def _tool_get_dataset_brief(args: dict, ctx: AgentContext) -> dict:
    """Return the compact dataset brief — schema, quality, signals, ML
    headlines, sample rows, AI narrative — all in one object.

    This is the right first stop for the agent when it needs general
    context. The fast Ask path injects it as a leading evidence block
    automatically, but the model can also call this tool explicitly when
    it wants to re-check the schema or pull a wider snapshot.
    """
    from modules.agent_brief import build_dataset_brief
    max_columns = int(args.get("max_columns") or 30)
    max_sample_rows = int(args.get("max_sample_rows") or 5)
    max_total_chars = int(args.get("max_total_chars") or 9000)
    brief = build_dataset_brief(
        ctx.result,
        max_columns=max_columns,
        max_sample_rows=max_sample_rows,
        max_total_chars=max_total_chars,
    )
    return {"ok": True, "brief": brief}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AGENT_TOOLS: dict[str, dict[str, Any]] = {
    "get_section": {
        "description": "Read a slice of the analysis result by dotted path (e.g. 'health.components.completeness').",
        "parameters_schema": {"path": "string"},
        "callable": _tool_get_section,
    },
    "list_columns": {
        "description": "List dataset columns with dtypes.",
        "parameters_schema": {},
        "callable": _tool_list_columns,
    },
    "column_summary": {
        "description": "Detailed summary for one column (missingness, dtype, descriptive stats, roles).",
        "parameters_schema": {"name": "string"},
        "callable": _tool_column_summary,
    },
    "run_query": {
        "description": (
            "Run a sandboxed pandas expression on `df`. "
            "Only safe pandas attributes are allowed; no imports / lambdas / IO."
        ),
        "parameters_schema": {"expression": "string"},
        "callable": _tool_run_query,
    },
    "get_top_anomalies": {
        "description": "Top-k anomalous rows from the anomaly ensemble.",
        "parameters_schema": {"k": "integer"},
        "callable": _tool_get_top_anomalies,
    },
    "get_leaderboard": {
        "description": "Model leaderboard winner + top rows.",
        "parameters_schema": {},
        "callable": _tool_get_leaderboard,
    },
    "get_explainability_top": {
        "description": "Top-k global SHAP features + a few per-row stories.",
        "parameters_schema": {"k": "integer"},
        "callable": _tool_get_explainability_top,
    },
    "search_facts": {
        "description": "RAG-style search over the analysis result for relevant facts.",
        "parameters_schema": {"question": "string", "k": "integer"},
        "callable": _tool_search_facts,
    },
    "get_dataset_brief": {
        "description": (
            "Return a compact JSON snapshot of the analyzed dataset: identity, schema, "
            "quality scores, signals, ML headlines, sample rows, and the AI narrative. "
            "Use this first to ground answers in concrete dataset facts."
        ),
        "parameters_schema": {
            "max_columns": "integer",
            "max_sample_rows": "integer",
            "max_total_chars": "integer",
        },
        "callable": _tool_get_dataset_brief,
    },
}


def list_tools() -> list[dict]:
    """Return a JSON-safe list of tool descriptors for prompt injection."""
    return [
        {
            "name": name,
            "description": meta["description"],
            "parameters": meta["parameters_schema"],
        }
        for name, meta in AGENT_TOOLS.items()
    ]


def run_tool(name: str, args: dict | None, ctx: AgentContext) -> dict:
    """Validate + dispatch a tool call. Always returns a dict."""
    meta = AGENT_TOOLS.get(name)
    if meta is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    fn: Callable[[dict, AgentContext], dict] = meta["callable"]
    try:
        return fn(args or {}, ctx)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
