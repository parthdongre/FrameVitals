"""Column-role inference for streaming dataset sources.

Semantic/value-pattern inference remains bounded, while full-source profile facts
correct the sample roles where the streaming profiler has authoritative or
full-stream sketch evidence. Every column records the scope of its cardinality
evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from framevitals.column_roles import (
    ID_KEYWORDS,
    _classify_missingness,
    _name_matches,
    infer_column_roles,
    summarize_roles,
)


_CARDINALITY_ROLES = {
    "constant",
    "binary",
    "low_cardinality",
    "high_cardinality",
    "unique_like",
}
_MISSINGNESS_ROLES = {
    "complete",
    "low_missing",
    "moderate_missing",
    "high_missing",
    "very_high_missing",
    "severe_missing",
}
_DERIVED_ROLES = {
    "analysis_candidate",
    "target_candidate",
    "regression_target_candidate",
}


def _categorical_cardinality(
    profile: Mapping[str, Any],
    column: str,
    *,
    sampled: bool,
) -> tuple[int | None, str, str, bool]:
    summaries = profile.get("categorical_summary", {})
    if not isinstance(summaries, Mapping):
        return None, "unavailable", "unavailable", True
    raw = summaries.get(column)
    if not isinstance(raw, Mapping) or "unique_values" not in raw:
        return None, "unavailable", "unavailable", True

    metadata = profile.get("categorical_summary_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    native_columns = set(metadata.get("native_full_stream_columns", []) or [])
    fallback_columns = set(metadata.get("sample_fallback_columns", []) or [])

    unique_count = int(raw.get("unique_values", 0) or 0)
    approximate = bool(raw.get("approximate", False))
    method = str(raw.get("unique_values_method") or metadata.get("method") or "streaming_profile")

    if column in native_columns:
        return unique_count, "full_stream_approximate", method, True
    if column in fallback_columns:
        scope = "bounded_row_sample" if sampled else "full_source"
        fallback_method = "evenly_spaced_row_sample" if sampled else "exact_full_source_sample"
        return unique_count, scope, fallback_method, sampled

    # A categorical result without explicit fallback metadata is exact only when
    # the retained sample covers the entire source. Otherwise keep it scoped as
    # approximate instead of silently promoting a sample statistic.
    scope = "bounded_row_sample" if sampled else "full_source"
    return unique_count, scope, method, approximate or sampled


def _apply_cardinality_roles(
    roles: set[str],
    *,
    column: str,
    unique_count: int,
    rows: int,
    is_numeric: bool,
    semantic_type: str | None,
) -> None:
    unique_ratio = unique_count / max(rows, 1)
    if unique_count <= 1:
        roles.add("constant")
    if unique_count == 2:
        roles.add("binary")
    if 2 <= unique_count <= 10:
        roles.add("low_cardinality")
    if unique_ratio > 0.80:
        roles.add("high_cardinality")
    if unique_ratio > 0.95:
        roles.add("unique_like")

    id_from_name = _name_matches(column, ID_KEYWORDS)
    if semantic_type == "uuid":
        roles.add("id_like")
    elif id_from_name:
        if not is_numeric or unique_ratio > 0.99:
            roles.add("id_like")
    elif unique_ratio > 0.95 and not is_numeric:
        roles.add("id_like")


def _recompute_derived_roles(
    roles: set[str],
    *,
    is_numeric: bool,
    unique_count: int,
) -> None:
    roles.difference_update(_DERIVED_ROLES)
    excluded = {"id_like", "time_like", "sequence_like", "constant"}
    if not roles.intersection(excluded):
        roles.add("analysis_candidate")

    if "id_like" not in roles and "constant" not in roles:
        if "binary" in roles or "low_cardinality" in roles:
            roles.add("target_candidate")
        elif is_numeric and unique_count > 10:
            roles.add("regression_target_candidate")


def infer_streaming_column_roles(
    sample: pd.DataFrame,
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Infer roles from a bounded sample corrected by full-stream profile facts."""
    source_rows = int(profile.get("shape", {}).get("rows", len(sample)) or len(sample))
    sample_rows = int(len(sample))
    sampled = sample_rows < source_rows
    missing_counts = profile.get("missing_counts", {})
    if not isinstance(missing_counts, Mapping):
        missing_counts = {}

    sample_roles = infer_column_roles(sample)
    columns: dict[str, Any] = {}

    for column, raw_info in sample_roles.items():
        info = dict(raw_info)
        roles = set(info.get("roles", []))
        roles.difference_update(_MISSINGNESS_ROLES)
        roles.difference_update(_CARDINALITY_ROLES)

        missing_count = int(missing_counts.get(column, sample[column].isna().sum()) or 0)
        non_missing = max(source_rows - missing_count, 0)
        missing_percent = round(missing_count / max(source_rows, 1) * 100, 2)
        roles.add(_classify_missingness(missing_percent))

        (
            categorical_unique,
            categorical_scope,
            categorical_method,
            categorical_approximate,
        ) = _categorical_cardinality(profile, column, sampled=sampled)
        if categorical_unique is not None:
            unique_count = categorical_unique
            cardinality_scope = categorical_scope
            cardinality_method = categorical_method
            cardinality_approximate = categorical_approximate
        else:
            unique_count = int(sample[column].nunique(dropna=True))
            cardinality_scope = "bounded_row_sample" if sampled else "full_source"
            cardinality_method = (
                "evenly_spaced_row_sample" if sampled else "exact_full_source_sample"
            )
            cardinality_approximate = sampled

        # Sample uniqueness may have introduced id_like. Remove that role unless
        # it is justified again by source-corrected cardinality/name/semantics.
        roles.discard("id_like")
        _apply_cardinality_roles(
            roles,
            column=str(column),
            unique_count=unique_count,
            rows=source_rows,
            is_numeric=bool(info.get("is_numeric")),
            semantic_type=info.get("semantic_type"),
        )
        _recompute_derived_roles(
            roles,
            is_numeric=bool(info.get("is_numeric")),
            unique_count=unique_count,
        )

        info.update({
            "roles": sorted(roles),
            "missing_percent": missing_percent,
            "non_missing_count": non_missing,
            "unique_count": unique_count,
            "unique_ratio": round(unique_count / max(source_rows, 1), 4),
            "cardinality_scope": cardinality_scope,
            "cardinality_method": cardinality_method,
            "cardinality_approximate": cardinality_approximate,
            "semantic_scope": "bounded_row_sample" if sampled else "full_source",
            "sample_rows": sample_rows,
            "source_rows": source_rows,
        })
        columns[str(column)] = info

    return {
        "columns": columns,
        "summary": summarize_roles(columns),
        "execution": {
            "method": "streaming_profile_with_bounded_semantic_sample",
            "full_materialization": False,
            "source_rows": source_rows,
            "sample_rows": sample_rows,
            "sampled": sampled,
            "full_source_inputs": ["missingness", "native_categorical_sketches"],
            "sample_inputs": [
                "semantic_patterns",
                "numeric_cardinality",
                "categorical_cardinality_without_native_sketch",
            ],
        },
    }
