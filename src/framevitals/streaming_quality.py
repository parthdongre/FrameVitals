"""Quality diagnostics for streaming dataset sources.

This adapter reuses FrameVitals' deterministic quality checks on a bounded row
sample while preserving full-source profile facts. Findings whose truth cannot
be proven from a sample (for example primary-key uniqueness or duplicate-column
identity) are explicitly reported as candidates rather than full-source facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from framevitals.column_roles import infer_column_roles
from framevitals.provenance import normalize_execution
from framevitals.quality_diagnostics import run_quality_diagnostics


_SAMPLE_FINDING_KEYS = (
    "identifier_duplicates",
    "quasi_constant_columns",
    "coercion_candidates",
    "category_normalisation",
    "blank_strings",
    "infinite_values",
    "mixed_object_types",
    "missingness_relationships",
)


def _annotate_sample_findings(
    payload: dict[str, Any],
    *,
    source_rows: int,
    sample_rows: int,
) -> None:
    sampled = sample_rows < source_rows
    if not sampled:
        return

    for key in _SAMPLE_FINDING_KEYS:
        findings = payload.get(key, [])
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            finding["sampled"] = True
            finding["sample_rows"] = sample_rows
            finding["source_rows"] = source_rows
            if key == "identifier_duplicates":
                finding["count_semantics"] = "lower_bound_from_sample"

    primary_keys = payload.get("primary_key_candidates", [])
    if isinstance(primary_keys, list):
        for finding in primary_keys:
            if not isinstance(finding, dict):
                continue
            finding["confidence"] = "candidate"
            finding["sampled"] = True
            finding["sample_rows"] = sample_rows
            finding["source_rows"] = source_rows
            finding["full_source_uniqueness_confirmed"] = False
            finding["reason"] = (
                "Column is complete and unique within the bounded row sample; "
                "full-source uniqueness was not confirmed."
            )

    duplicate_columns = payload.get("duplicate_columns", [])
    if isinstance(duplicate_columns, list):
        for finding in duplicate_columns:
            if not isinstance(finding, dict):
                continue
            finding["sampled"] = True
            finding["sample_rows"] = sample_rows
            finding["source_rows"] = source_rows
            finding["confirmed_with_full_equality"] = False
            finding["candidate_only"] = True
            finding["confirmation_scope"] = "bounded_row_sample"


def run_streaming_quality_diagnostics(
    sample: pd.DataFrame,
    *,
    profile: Mapping[str, Any],
    source_rows: int,
    source_columns: int,
    max_sample_rows: int = 5_000,
    max_columns: int = 100,
    max_missingness_columns: int = 25,
) -> dict[str, Any]:
    """Run quality diagnostics without materializing a streaming source.

    Full-source missingness and duplicate-row estimates come from ``profile``.
    Value-level diagnostics operate on ``sample``. The returned execution
    metadata describes that split so downstream consumers can distinguish
    observed facts from sample-derived candidates.
    """
    if source_rows < 1:
        raise ValueError("source_rows must be at least 1.")
    if source_columns < 1:
        raise ValueError("source_columns must be at least 1.")

    roles = infer_column_roles(sample)
    payload = run_quality_diagnostics(
        sample,
        profile=profile,
        column_roles=roles,
        max_sample_rows=max_sample_rows,
        max_columns=max_columns,
        max_missingness_columns=max_missingness_columns,
    )

    sample_rows = int(len(sample))
    _annotate_sample_findings(
        payload,
        source_rows=int(source_rows),
        sample_rows=sample_rows,
    )

    payload["rows"] = int(source_rows)
    payload["columns"] = int(source_columns)
    payload["columns_checked"] = min(int(source_columns), int(max_columns))
    payload["truncated_columns"] = int(source_columns) > int(max_columns)
    payload["duplicate_rows"] = int(profile.get("duplicate_rows", 0) or 0)

    issue_groups = (
        "identifier_duplicates",
        "quasi_constant_columns",
        "duplicate_columns",
        "coercion_candidates",
        "category_normalisation",
        "blank_strings",
        "infinite_values",
        "mixed_object_types",
        "missingness_relationships",
    )
    issue_count = sum(
        len(payload.get(key, []))
        for key in issue_groups
        if isinstance(payload.get(key, []), list)
    )
    duplicate_rows = int(payload["duplicate_rows"])
    payload["summary"] = {
        "issue_groups": sum(bool(payload.get(key)) for key in issue_groups),
        "issue_count": issue_count + (1 if duplicate_rows else 0),
        "primary_key_candidate_count": len(payload.get("primary_key_candidates", [])),
    }
    payload["execution"] = normalize_execution(
        {
            "method": "streaming_profile_with_bounded_quality_sample",
            "full_materialization": False,
            "source_rows": int(source_rows),
            "source_columns": int(source_columns),
            "sample_rows": sample_rows,
            "sampled": sample_rows < int(source_rows),
            "full_source_inputs": ["missingness", "duplicate_row_estimate"],
            "sample_inputs": [
                "column_roles",
                "identifier_duplicates",
                "quasi_constants",
                "duplicate_column_candidates",
                "coercion_candidates",
                "category_normalisation",
                "blank_strings",
                "infinite_values",
                "mixed_object_types",
                "missingness_relationships",
            ],
            "candidate_only_checks": (
                ["primary_key_candidates", "duplicate_columns"]
                if sample_rows < int(source_rows)
                else []
            ),
        },
        method="streaming_profile_with_bounded_quality_sample",
        full_materialization=False,
    )
    return payload
