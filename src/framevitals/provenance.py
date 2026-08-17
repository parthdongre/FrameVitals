"""Shared execution-provenance helpers for public FrameVitals results.

The 0.x API historically grew execution metadata inside individual diagnostics.
This module defines a small additive contract so those payloads can converge
without breaking callers that still rely on legacy fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


EXECUTION_SCHEMA_VERSION = "1"


def load_fully_materializes(metadata: Any) -> bool:
    """Return whether ``source.load()`` creates a complete pandas representation.

    A pandas DataFrame is already materialized before FrameVitals sees it. File,
    Arrow, relation, remote, and other source types may all require a complete
    conversion when an exact operation calls ``load()``; source kind alone is
    therefore not a reliable materialization signal.
    """
    return not (
        getattr(metadata, "kind", None) == "memory"
        and getattr(metadata, "format", None) == "pandas"
    )


def execution_provenance(
    method: str,
    *,
    full_materialization: bool,
    source: Mapping[str, Any] | None = None,
    sampled: bool | None = None,
    source_rows: int | None = None,
    source_columns: int | None = None,
    sample_rows: int | None = None,
    strategy: str | None = None,
    components: Mapping[str, Any] | None = None,
    reason: str | None = None,
    scope: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly execution block using the shared v1 vocabulary.

    Optional values are omitted rather than serialized as ``null`` so existing
    result payloads can adopt the schema incrementally. ``extra`` exists for
    operation-specific fields such as scale class, pair budget, or projection
    counts without making them part of the universal contract.
    """
    if not str(method).strip():
        raise ValueError("execution provenance method must not be empty.")

    payload: dict[str, Any] = {
        "execution_schema_version": EXECUTION_SCHEMA_VERSION,
        "method": str(method),
        "full_materialization": bool(full_materialization),
    }
    optional = {
        "source": dict(source) if source is not None else None,
        "sampled": bool(sampled) if sampled is not None else None,
        "source_rows": int(source_rows) if source_rows is not None else None,
        "source_columns": int(source_columns) if source_columns is not None else None,
        "sample_rows": int(sample_rows) if sample_rows is not None else None,
        "strategy": strategy,
        "components": dict(components) if components is not None else None,
        "reason": reason,
        "scope": scope,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    if extra:
        payload.update(dict(extra))
    return payload


def normalize_execution(
    execution: Mapping[str, Any],
    *,
    method: str | None = None,
    full_materialization: bool | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add the shared v1 contract to an existing legacy execution mapping.

    Existing keys always win unless the caller explicitly supplies a method,
    materialization flag, or source. This makes migration additive during 0.x.
    """
    result = dict(execution)
    result["execution_schema_version"] = EXECUTION_SCHEMA_VERSION

    resolved_method = method or result.get("method") or result.get("scope")
    if resolved_method is not None:
        result["method"] = str(resolved_method)
    if full_materialization is not None:
        result["full_materialization"] = bool(full_materialization)
    else:
        result.setdefault("full_materialization", False)
    if source is not None:
        result["source"] = dict(source)
    return result
