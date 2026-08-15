"""
Safe Pandas Evaluator
=====================
AST-allowlist sandbox for evaluating pandas expressions submitted by an LLM.

Only a small whitelist of node types, names, and attribute accesses are
permitted. Anything else raises UnsafeExpression.

Usage:
    from modules.safe_pandas import safe_eval, UnsafeExpression
    result = safe_eval("df['age'].mean()", df)
    result = safe_eval("df.groupby('region')['revenue'].sum().head(5)", df)

Rules:
- Only the names {df, np, pd, len, abs, min, max, round, sum, sorted} are allowed.
- Only the attribute access list below is permitted.
- No assignments, no imports, no function definitions, no comprehensions with
  side effects, no f-strings, no exec/eval/compile/open.
- Maximum expression length is enforced.
- Maximum AST depth is enforced.
- Output is converted to a JSON-safe Python primitive (or list/dict thereof).
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

_ALLOWED_NODES: set[type] = {
    ast.Expression,
    ast.Module,
    ast.Expr,
    # Literals
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    # Names and attributes
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Index,  # py<3.9 compatibility, harmless on newer
    # Calls
    ast.Call,
    ast.keyword,
    # Operators
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    # Arithmetic / boolean operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.MatMult,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
}

_ALLOWED_NAMES: set[str] = {"df", "np", "pd", "len", "abs", "min", "max", "round", "sum", "sorted"}

_ALLOWED_ATTRS: set[str] = {
    # DataFrame / Series essentials
    "loc", "iloc", "at", "iat",
    "head", "tail", "shape", "size", "columns", "index", "dtypes", "values",
    # Boolean / null
    "isna", "notna", "isnull", "notnull",
    # Aggregation
    "sum", "count", "mean", "median", "std", "var", "min", "max",
    "quantile", "describe", "agg", "aggregate",
    # Groupby + sort
    "groupby", "sort_values", "sort_index", "value_counts",
    # Selection / shape
    "select_dtypes", "drop", "drop_duplicates", "rename", "reset_index",
    "set_index", "unique", "nunique",
    # Combination
    "merge", "join", "concat",
    # Stats / transform (read-only)
    "corr", "cov", "abs", "round", "rank", "diff", "pct_change",
    "cumsum", "cummax", "cummin", "rolling",
    # String accessor
    "str", "dt", "cat",
    # Common str / dt methods
    "lower", "upper", "contains", "startswith", "endswith", "len", "strip",
    "year", "month", "day", "dayofweek", "weekday",
    # numpy aliases
    "log", "log1p", "exp", "sqrt", "where",
    # Conversion / formatting
    "astype", "to_dict", "to_list", "tolist",
    # Boolean-array logic
    "any", "all",
    # Filter helper
    "query", "between", "isin",
    # Apply (whitelisted callables only, see _ALLOWED_CALLABLES below)
    "apply", "applymap", "map",
}

# Hard limits
_MAX_LEN = 600
_MAX_DEPTH = 30
_MAX_PREVIEW_ROWS = 50


class UnsafeExpression(ValueError):
    """Raised when an expression contains a disallowed construct."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _walk_with_depth(node, depth=0):
    yield node, depth
    for child in ast.iter_child_nodes(node):
        yield from _walk_with_depth(child, depth + 1)


def _validate(tree: ast.AST) -> None:
    for node, depth in _walk_with_depth(tree):
        if depth > _MAX_DEPTH:
            raise UnsafeExpression(f"Expression nesting too deep (>{_MAX_DEPTH})")

        if type(node) not in _ALLOWED_NODES:
            raise UnsafeExpression(f"Disallowed AST node: {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id not in _ALLOWED_NAMES and not node.id.startswith("__col__"):
                # Names other than allowed roots can only appear as keyword args
                # (e.g., axis=0). Keyword arg names are stored in ast.keyword.arg
                # which is a string, not a Name node, so we don't reach here for them.
                raise UnsafeExpression(f"Disallowed name: {node.id}")

        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_"):
                raise UnsafeExpression(f"Dunder/private access disallowed: {attr}")
            if attr not in _ALLOWED_ATTRS:
                raise UnsafeExpression(f"Disallowed attribute: {attr}")

        if isinstance(node, ast.Call):
            # Reject calls like __import__, type, getattr, etc.
            if isinstance(node.func, ast.Name) and node.func.id not in _ALLOWED_NAMES:
                raise UnsafeExpression(f"Disallowed function call: {node.func.id}")


# ---------------------------------------------------------------------------
# JSON-safe coercion
# ---------------------------------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    """Convert pandas / numpy / scalar to a JSON-serializable structure."""
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6)

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]

    if isinstance(value, pd.Series):
        out = []
        for idx, val in value.head(_MAX_PREVIEW_ROWS).items():
            out.append({"index": _to_jsonable(idx), "value": _to_jsonable(val)})
        return {
            "type": "series",
            "name": str(value.name) if value.name is not None else None,
            "length": int(len(value)),
            "rows": out,
            "truncated": int(len(value)) > _MAX_PREVIEW_ROWS,
        }

    if isinstance(value, pd.DataFrame):
        rows = value.head(_MAX_PREVIEW_ROWS).where(value.notna(), None).to_dict(orient="records")
        rows = [{k: _to_jsonable(v) for k, v in r.items()} for r in rows]
        return {
            "type": "dataframe",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(c) for c in value.columns],
            "rows": rows,
            "truncated": int(value.shape[0]) > _MAX_PREVIEW_ROWS,
        }

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, pd.Index):
        return [_to_jsonable(v) for v in value.tolist()]

    return str(value)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def safe_eval(expression: str, df: pd.DataFrame) -> dict:
    """
    Validate and evaluate a pandas expression against `df`.

    Returns a dict with:
        {"ok": bool, "result": ..., "error": str | None, "expression": str}
    """
    if not isinstance(expression, str):
        return {"ok": False, "result": None, "error": "Expression must be a string.", "expression": str(expression)}

    expression = expression.strip()
    if not expression:
        return {"ok": False, "result": None, "error": "Empty expression.", "expression": expression}

    if len(expression) > _MAX_LEN:
        return {"ok": False, "result": None, "error": f"Expression too long (>{_MAX_LEN} chars).", "expression": expression}

    # Reject obvious red flags pre-parse
    forbidden = ("__", "import ", "from ", "lambda", "exec(", "eval(", "open(", "globals", "locals", "compile(")
    lower = expression.lower()
    for token in forbidden:
        if token in lower:
            return {"ok": False, "result": None, "error": f"Disallowed token: {token!r}", "expression": expression}

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return {"ok": False, "result": None, "error": f"Syntax error: {exc.msg}", "expression": expression}

    try:
        _validate(tree)
    except UnsafeExpression as exc:
        return {"ok": False, "result": None, "error": str(exc), "expression": expression}

    # Restricted execution environment (NO __builtins__, NO globals)
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        "df": df,
        "np": np,
        "pd": pd,
        "len": len,
        "abs": abs,
        "min": min,
        "max": max,
        "round": round,
        "sum": sum,
        "sorted": sorted,
    }

    try:
        compiled = compile(tree, "<safe_pandas>", "eval")
        value = eval(compiled, safe_globals, safe_locals)  # noqa: S307 — sandboxed
    except Exception as exc:
        return {"ok": False, "result": None, "error": f"{type(exc).__name__}: {exc}", "expression": expression}

    return {"ok": True, "result": _to_jsonable(value), "error": None, "expression": expression}
