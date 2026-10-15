"""Strict, shared JSON conversion for results and transport boundaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    # NumPy scalar support without eagerly importing NumPy.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def dumps(value: Any, *, indent: int = 2, **kwargs: Any) -> str:
    kwargs.pop("default", None)
    kwargs.pop("allow_nan", None)
    return json.dumps(json_safe(value), indent=indent, allow_nan=False, **kwargs)
