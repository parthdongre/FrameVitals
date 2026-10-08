"""Diagnostics for related collections of tabular datasets."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from framevitals.core.beacons import beacon
from framevitals.quality_results import DriftResult
from framevitals.result import AnalysisResult


def _to_pandas(value: Any, *, max_rows: int) -> pd.DataFrame:
    module = type(value).__module__
    if module.startswith("pandas."):
        frame = value
    elif callable(getattr(value, "to_pandas", None)):
        try:
            frame = value.to_pandas()
        except Exception:
            if callable(getattr(value, "to_dicts", None)):
                frame = pd.DataFrame(value.to_dicts())
            else:
                raise
    else:
        frame = pd.DataFrame(value)

    if len(frame) > max_rows:
        return frame.iloc[:max_rows].copy()
    return frame.copy()


def _key_candidates(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    rows = len(frame)
    if rows == 0:
        return candidates
    for column in frame.columns:
        series = frame[column]
        if series.isna().any():
            continue
        try:
            unique = int(series.nunique(dropna=False))
        except Exception:
            continue
        if unique == rows:
            candidates.append(str(column))
    return candidates


def _table_summary(frame: pd.DataFrame, *, source_rows: int) -> dict[str, Any]:
    missing = {
        str(column): round(float(frame[column].isna().mean()), 8)
        for column in frame.columns
    }
    return {
        "rows": int(source_rows),
        "sample_rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_fraction": missing,
        "duplicate_rows_sample": int(frame.duplicated().sum()),
        "key_candidates": _key_candidates(frame),
    }


def _relationship(
    left_name: str,
    left: pd.DataFrame,
    left_summary: dict[str, Any],
    right_name: str,
    right: pd.DataFrame,
    right_summary: dict[str, Any],
    column: str,
) -> dict[str, Any]:
    left_series = left[column].dropna()
    right_series = right[column].dropna()
    left_unique = column in left_summary["key_candidates"]
    right_unique = column in right_summary["key_candidates"]

    left_values = set(left_series.astype(str).tolist())
    right_values = set(right_series.astype(str).tolist())
    intersection = left_values & right_values
    left_coverage = len(intersection) / max(1, len(left_values))
    right_coverage = len(intersection) / max(1, len(right_values))

    if left_unique and right_unique:
        cardinality = "one_to_one"
        parent = None
        child = None
        child_coverage = min(left_coverage, right_coverage)
    elif left_unique:
        cardinality = "one_to_many"
        parent = left_name
        child = right_name
        child_coverage = right_coverage
    elif right_unique:
        cardinality = "many_to_one"
        parent = right_name
        child = left_name
        child_coverage = left_coverage
    else:
        cardinality = "many_to_many"
        parent = None
        child = None
        child_coverage = min(left_coverage, right_coverage)

    left_counts = left_series.astype(str).value_counts()
    right_counts = right_series.astype(str).value_counts()
    common = set(left_counts.index) & set(right_counts.index)
    estimated_join_rows_sample = int(
        sum(int(left_counts[key]) * int(right_counts[key]) for key in common)
    )
    join_expansion = estimated_join_rows_sample / max(1, len(left), len(right))

    return {
        "left_table": left_name,
        "right_table": right_name,
        "column": column,
        "cardinality": cardinality,
        "parent_table": parent,
        "child_table": child,
        "left_unique": bool(left_unique),
        "right_unique": bool(right_unique),
        "left_value_coverage": round(left_coverage, 8),
        "right_value_coverage": round(right_coverage, 8),
        "child_reference_coverage": round(child_coverage, 8),
        "estimated_join_rows_sample": estimated_join_rows_sample,
        "join_expansion_ratio_sample": round(float(join_expansion), 8),
    }


def inspect_relational(
    project: dict[str, Any],
    *,
    max_rows_per_table: int = 100_000,
    max_relationship_columns: int = 20,
) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}

    for name, source in project.items():
        table_name = str(name)
        source_rows = len(source) if hasattr(source, "__len__") else 0
        frame = _to_pandas(source, max_rows=max_rows_per_table)
        frames[table_name] = frame
        summaries[table_name] = _table_summary(frame, source_rows=int(source_rows))

    relationships: list[dict[str, Any]] = []
    for left_name, right_name in combinations(sorted(frames), 2):
        left = frames[left_name]
        right = frames[right_name]
        common = [
            str(column)
            for column in left.columns
            if str(column) in {str(value) for value in right.columns}
        ]
        common.sort(
            key=lambda name: (
                0 if name.lower() == "id" or name.lower().endswith("_id") else 1,
                name,
            )
        )
        for column in common[:max_relationship_columns]:
            if column not in left.columns or column not in right.columns:
                continue
            relationships.append(
                _relationship(
                    left_name,
                    left,
                    summaries[left_name],
                    right_name,
                    right,
                    summaries[right_name],
                    column,
                )
            )

    linked_tables = {
        relationship["left_table"]
        for relationship in relationships
    } | {
        relationship["right_table"]
        for relationship in relationships
    }

    return {
        "table_count": len(frames),
        "tables": summaries,
        "relationships": relationships,
        "relationship_count": len(relationships),
        "isolated_tables": sorted(set(frames) - linked_tables),
        "sampled": any(
            summary["sample_rows"] < summary["rows"]
            for summary in summaries.values()
        ),
        "max_rows_per_table": int(max_rows_per_table),
    }


def _health_label(score: float) -> str:
    if score >= 90:
        return "healthy"
    if score >= 75:
        return "good"
    if score >= 55:
        return "attention"
    return "critical"


def analyze_relational(
    project: dict[str, Any],
    *,
    depth: str | None = None,
) -> AnalysisResult:
    mode = str(depth or "standard").lower()
    max_rows = {
        "quick": 20_000,
        "standard": 100_000,
        "deep": 300_000,
        "research": 1_000_000,
    }.get(mode, 100_000)

    summary = inspect_relational(project, max_rows_per_table=max_rows)
    findings: list[dict[str, Any]] = []

    weak_references = [
        item
        for item in summary["relationships"]
        if item["parent_table"]
        and float(item["child_reference_coverage"]) < 0.98
    ]
    if weak_references:
        findings.append(beacon(
            "relational.referential_gaps",
            "Potential referential-integrity gaps were detected",
            severity="high",
            confidence=0.9,
            summary=f"{len(weak_references)} inferred relationships have <98% child-key coverage.",
            recommendation="Inspect orphan foreign-key values and upstream entity synchronization.",
            evidence={"relationships": weak_references[:30]},
        ))

    many_to_many = [
        item
        for item in summary["relationships"]
        if item["cardinality"] == "many_to_many"
        and float(item["join_expansion_ratio_sample"]) >= 2.0
    ]
    if many_to_many:
        findings.append(beacon(
            "relational.join_explosion",
            "Potential many-to-many join expansion was detected",
            severity="medium",
            confidence=0.9,
            summary=f"{len(many_to_many)} shared columns can multiply rows materially when joined.",
            recommendation="Confirm intended join keys and pre-aggregate or deduplicate where appropriate.",
            evidence={"relationships": many_to_many[:30]},
        ))

    duplicate_key_tables = []
    for table_name, table in summary["tables"].items():
        if not table["key_candidates"] and table["rows"] > 0:
            id_like = [
                name for name in table["column_names"]
                if name.lower() == "id" or name.lower().endswith("_id")
            ]
            if id_like:
                duplicate_key_tables.append({
                    "table": table_name,
                    "id_like_columns": id_like,
                })
    if duplicate_key_tables:
        findings.append(beacon(
            "relational.no_unique_id",
            "Some tables have ID-like columns but no unique key candidate",
            severity="medium",
            confidence=0.85,
            summary=f"{len(duplicate_key_tables)} tables may have duplicated entity identifiers.",
            recommendation="Validate primary-key uniqueness before joins or downstream modeling.",
            evidence={"tables": duplicate_key_tables[:30]},
        ))

    if summary["isolated_tables"] and summary["table_count"] > 1:
        findings.append(beacon(
            "relational.isolated_tables",
            "Some tables have no inferred shared-key relationship",
            severity="info",
            confidence=0.8,
            summary=f"{len(summary['isolated_tables'])} tables are isolated in the inferred schema graph.",
            recommendation="Confirm whether these tables are intentionally independent or missing join keys.",
            evidence={"tables": summary["isolated_tables"]},
        ))

    score = 100.0
    score -= min(40.0, len(weak_references) * 8.0)
    score -= min(20.0, len(many_to_many) * 4.0)
    score -= min(15.0, len(duplicate_key_tables) * 3.0)
    score = round(max(0.0, min(100.0, score)), 2)

    total_rows = sum(int(table["rows"]) for table in summary["tables"].values())
    total_columns = sum(int(table["columns"]) for table in summary["tables"].values())

    return AnalysisResult({
        "dataset_id": None,
        "filename": "relational_project",
        "analysis_mode": mode,
        "source_kind": "relational",
        "profile": {
            "shape": {"rows": total_rows, "columns": total_columns},
            "structure": summary,
        },
        "relational": summary,
        "health": {"overall_score": score, "label": _health_label(score)},
        "ml_readiness": {"score": None, "label": "not_applicable"},
        "findings": findings,
        "artifacts_enabled": False,
        "execution": {
            "method": "bounded_relational_diagnostics",
            "sampled": bool(summary["sampled"]),
            "resource_bounded": True,
        },
    })


def compare_relational(reference: dict[str, Any], current: dict[str, Any]) -> DriftResult:
    ref = inspect_relational(reference, max_rows_per_table=100_000)
    cur = inspect_relational(current, max_rows_per_table=100_000)

    ref_tables = set(ref["tables"])
    cur_tables = set(cur["tables"])
    common = sorted(ref_tables & cur_tables)

    row_changes = {}
    column_changes = {}
    for table in common:
        ref_table = ref["tables"][table]
        cur_table = cur["tables"][table]
        before_rows = int(ref_table["rows"])
        after_rows = int(cur_table["rows"])
        if before_rows != after_rows:
            row_changes[table] = {
                "reference": before_rows,
                "current": after_rows,
                "ratio": round(abs(after_rows - before_rows) / max(1, before_rows), 8),
            }

        ref_cols = set(ref_table["column_names"])
        cur_cols = set(cur_table["column_names"])
        if ref_cols != cur_cols:
            column_changes[table] = {
                "added": sorted(cur_cols - ref_cols),
                "removed": sorted(ref_cols - cur_cols),
            }

    ref_relationships = {
        (item["left_table"], item["right_table"], item["column"], item["cardinality"])
        for item in ref["relationships"]
    }
    cur_relationships = {
        (item["left_table"], item["right_table"], item["column"], item["cardinality"])
        for item in cur["relationships"]
    }

    table_churn = len(ref_tables ^ cur_tables) / max(1, len(ref_tables | cur_tables))
    max_row_change = max(
        (float(item["ratio"]) for item in row_changes.values()),
        default=0.0,
    )
    relationship_churn = len(ref_relationships ^ cur_relationships) / max(
        1,
        len(ref_relationships | cur_relationships),
    )
    column_churn = len(column_changes) / max(1, len(common))

    score = max(
        min(1.0, table_churn * 2.0),
        min(1.0, max_row_change),
        min(1.0, relationship_churn),
        min(1.0, column_churn),
    )
    severity = "severe" if score >= 0.75 else "moderate" if score >= 0.40 else "minor" if score >= 0.15 else "stable"
    status = "fail" if severity == "severe" else "warn" if severity != "stable" else "pass"

    return DriftResult({
        "available": True,
        "source_kind": "relational",
        "gate": {"status": status, "severity": severity},
        "summary": {"overall_verdict": severity, "change_score": round(score, 6)},
        "relational": {
            "reference_table_count": len(ref_tables),
            "current_table_count": len(cur_tables),
            "added_tables": sorted(cur_tables - ref_tables),
            "removed_tables": sorted(ref_tables - cur_tables),
            "row_changes": row_changes,
            "column_changes": column_changes,
            "relationship_changes": {
                "added": [list(item) for item in sorted(cur_relationships - ref_relationships)],
                "removed": [list(item) for item in sorted(ref_relationships - cur_relationships)],
            },
        },
        "columns": [],
    })
