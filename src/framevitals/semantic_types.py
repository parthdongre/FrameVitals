"""Deterministic semantic type inference for text-like columns.

The detector intentionally uses bounded samples so semantic typing remains
cheap on large datasets. It augments, rather than replaces, dtype and name-based
column role inference.
"""

from __future__ import annotations

import ipaddress
import json
import re
import uuid
from typing import Callable

import pandas as pd


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE = re.compile(r"^(?:https?://|www\.)[^\s]+$", re.IGNORECASE)
_PHONE_RE = re.compile(r"^\+?[0-9][0-9\s().-]{6,}$")
_PERCENT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*%$")
_CURRENCY_RE = re.compile(
    r"^(?:[$€£₹¥]\s*[+-]?[\d,.]+|(?:USD|EUR|GBP|INR|JPY)\s+[+-]?[\d,.]+)$",
    re.IGNORECASE,
)
_BOOL_TOKEN_RE = re.compile(
    r"^(?:true|false|yes|no|y|n|on|off)$",
    re.IGNORECASE,
)


def _matches_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _matches_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _matches_phone(value: str) -> bool:
    if not _PHONE_RE.match(value):
        return False
    digit_count = sum(character.isdigit() for character in value)
    return 7 <= digit_count <= 15


def _matches_json(value: str) -> bool:
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, (dict, list))


def _ratio(sample: list[str], predicate: Callable[[str], bool]) -> float:
    if not sample:
        return 0.0
    return sum(1 for value in sample if predicate(value)) / len(sample)


def infer_semantic_types(
    series: pd.Series,
    *,
    max_samples: int = 100,
    threshold: float = 0.70,
) -> dict:
    """Infer semantic value types from a bounded non-null sample.

    Returns a dictionary with ``primary``, ranked ``candidates``, and
    ``sample_size``. Empty/unsupported columns return no candidates.
    """
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in the interval (0, 1].")

    is_text = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    if not is_text:
        return {"primary": None, "candidates": [], "sample_size": 0}

    sample = [
        value.strip()
        for value in series.dropna().astype(str).head(max_samples).tolist()
        if value.strip()
    ]
    if not sample:
        return {"primary": None, "candidates": [], "sample_size": 0}

    checks: list[tuple[str, Callable[[str], bool]]] = [
        ("email", lambda value: bool(_EMAIL_RE.match(value))),
        ("url", lambda value: bool(_URL_RE.match(value))),
        ("uuid", _matches_uuid),
        ("ip_address", _matches_ip),
        ("phone", _matches_phone),
        ("percentage", lambda value: bool(_PERCENT_RE.match(value))),
        ("currency", lambda value: bool(_CURRENCY_RE.match(value))),
        ("json", _matches_json),
        ("boolean_token", lambda value: bool(_BOOL_TOKEN_RE.match(value))),
    ]

    candidates = []
    for semantic_type, predicate in checks:
        confidence = _ratio(sample, predicate)
        if confidence >= threshold:
            candidates.append({
                "type": semantic_type,
                "confidence": round(confidence, 4),
            })

    candidates.sort(key=lambda item: (-item["confidence"], item["type"]))
    return {
        "primary": candidates[0]["type"] if candidates else None,
        "candidates": candidates,
        "sample_size": len(sample),
    }
