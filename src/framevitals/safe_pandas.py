"""Safe, read-only pandas expression evaluator for the optional agent layer.

Expressions are parsed with an AST allowlist and evaluated against an isolated
DataFrame copy. The evaluator intentionally exposes a small analytical surface;
private attributes, imports, arbitrary code execution, pandas ``query`` strings,
and mutating access to the caller's DataFrame are not permitted.
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pandas as pd


_ALLOWED_NODES: set[type] = {
    ast.Expression,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Call,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
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

_ALLOWED_NAMES: set[str] = {
    "df",
    "np",
    "pd",
    "len",
    "abs",
    "min",
    "max",
    "round",
    "sum",
    "sorted",
}

_ALLOWED_ATTRS: set[str] = {
    "loc",
    "iloc",
    "at",
    "iat",
    "head",
    "tail",
    "shape",
    "size",
    "columns",
    "index",
    "dtypes",
    "values",
    "isna",
    "notna",
    "isnull",
    "notnull",
    "sum",
    "count",
    "mean",
    "median",
    "std",
    "var",
    "min",
    "max",
    "quantile",
    "describe",
    "agg",
    "aggregate",
    "groupby",
    "sort_values",
    "sort_index",
    "value_counts",
    "select_dtypes",
    "drop",
    "drop_duplicates",
    "rename",
    "reset_index",
    "set_index",
    "unique",
    "nunique",
    "merge",
    "join",
    "concat",
    "corr",
    "cov",
    "abs",
    "round",
    "rank",
    "diff",
    "pct_change",
    "cumsum",
    "cummax",
    "cummin",
    "rolling",
    "str",
    "dt",
    "cat",
    "lower",
    "upper",
    "contains",
    "startswith",
    "endswith",
    "len",
    "strip",
    "year",
    "month",
    "day",
    "dayofweek",
    "weekday",
    "log",
    "log1p",
    "exp",
    "sqrt",
    "where",
    "astype",
    "to_dict",
    "to_list",
    "tolist",
    "any",
    "all",
    "between",
    "isin",
    "apply",
    "map",
}

_MAX_LEN = 600
_MAX_DEPTH = 30
_MAX_PREVIEW_ROWS = 50


class UnsafeExpression(ValueError):
    """Raised when an expression contains a disallowed construct."""


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
                raise UnsafeExpression(f"Disallowed name: {node.id}")

        if isinstance(node, ast.Attribute):
            attribute = node.attr
            if attribute.startswith("_"):
                raise UnsafeExpression(
                    f"Dunder/private access disallowed: {attribute}"
                )
            if attribute not in _ALLOWED_ATTRS:
                raise UnsafeExpression(f"Disallowed attribute: {attribute}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in _ALLOWED_NAMES:
                raise UnsafeExpression(f"Disallowed function call: {node.func.id}")


def _to_jsonable(value: Any) -> Any:
    """Convert pandas/numpy values into a bounded JSON-serializable structure."""
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        normalized = float(value)
        if math.isnan(normalized) or math.isinf(normalized):
            return None
        return round(normalized, 6)
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]

    if isinstance(value, pd.Series):
        rows = [
            {"index": _to_jsonable(index), "value": _to_jsonable(item)}
            for index, item in value.head(_MAX_PREVIEW_ROWS).items()
        ]
        return {
            "type": "series",
            "name": str(value.name) if value.name is not None else None,
            "length": int(len(value)),
            "rows": rows,
            "truncated": int(len(value)) > _MAX_PREVIEW_ROWS,
        }

    if isinstance(value, pd.DataFrame):
        records = (
            value.head(_MAX_PREVIEW_ROWS)
            .where(value.head(_MAX_PREVIEW_ROWS).notna(), None)
            .to_dict(orient="records")
        )
        rows = [
            {key: _to_jsonable(item) for key, item in record.items()}
            for record in records
        ]
        return {
            "type": "dataframe",
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(column) for column in value.columns],
            "rows": rows,
            "truncated": int(value.shape[0]) > _MAX_PREVIEW_ROWS,
        }

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, pd.Index):
        return [_to_jsonable(item) for item in value.tolist()]
    return str(value)


def safe_eval(expression: str, df: pd.DataFrame) -> dict:
    """Validate and evaluate a read-only pandas expression against ``df``."""
    if not isinstance(expression, str):
        return {
            "ok": False,
            "result": None,
            "error": "Expression must be a string.",
            "expression": str(expression),
        }
    if not isinstance(df, pd.DataFrame):
        return {
            "ok": False,
            "result": None,
            "error": "df must be a pandas DataFrame.",
            "expression": expression,
        }

    expression = expression.strip()
    if not expression:
        return {
            "ok": False,
            "result": None,
            "error": "Empty expression.",
            "expression": expression,
        }
    if len(expression) > _MAX_LEN:
        return {
            "ok": False,
            "result": None,
            "error": f"Expression too long (>{_MAX_LEN} chars).",
            "expression": expression,
        }

    forbidden = (
        "__",
        "import ",
        "from ",
        "lambda",
        "exec(",
        "eval(",
        "open(",
        "globals",
        "locals",
        "compile(",
        ".query(",
        ".applymap(",
    )
    lower = expression.lower()
    for token in forbidden:
        if token in lower:
            return {
                "ok": False,
                "result": None,
                "error": f"Disallowed token: {token!r}",
                "expression": expression,
            }

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return {
            "ok": False,
            "result": None,
            "error": f"Syntax error: {exc.msg}",
            "expression": expression,
        }

    try:
        _validate(tree)
    except UnsafeExpression as exc:
        return {
            "ok": False,
            "result": None,
            "error": str(exc),
            "expression": expression,
        }

    safe_globals = {"__builtins__": {}}
    safe_locals = {
        # Evaluate against an isolated copy so otherwise-useful pandas methods
        # cannot mutate the caller's data through ``inplace=True``.
        "df": df.copy(deep=True),
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
        value = eval(compiled, safe_globals, safe_locals)  # noqa: S307 - AST sandboxed
    except Exception as exc:
        return {
            "ok": False,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "expression": expression,
        }

    return {
        "ok": True,
        "result": _to_jsonable(value),
        "error": None,
        "expression": expression,
    }
