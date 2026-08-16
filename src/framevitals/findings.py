"""Normalized findings for FrameVitals analysis results.

FrameVitals diagnostics originate in several deterministic analysis layers. This
module translates their already-computed warnings/signals into one stable,
machine-readable finding shape without changing diagnostic thresholds.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


_SEVERITY_MAP = {
    "high": "high",
    "critical": "critical",
    "medium": "medium",
    "moderate": "medium",
    "low": "low",
    "informational": "info",
    "info": "info",
    "none": "none",
}

_SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    "none": 5,
}


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "finding"


def normalize_severity(value: Any) -> str:
    """Return a stable lower-case severity label."""
    if value is None:
        return "none"
    text = str(value).strip().lower()
    return _SEVERITY_MAP.get(text, text or "none")


def _sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings.sort(
        key=lambda item: (
            _SEVERITY_ORDER.get(str(item.get("severity")), 99),
            str(item.get("code", "")),
        )
    )
    return findings


def _is_actionable_signal(signal: dict[str, Any]) -> bool:
    status = str(signal.get("status", "")).strip().lower()
    name = str(signal.get("name", "")).strip().lower()

    if status == "review":
        return True
    if name == "ml readiness" and status not in {"ready", "good"}:
        return True
    return False


def findings_from_signals(signals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize existing display signals into actionable findings."""
    findings: list[dict[str, Any]] = []

    for signal in signals:
        if not isinstance(signal, dict) or not _is_actionable_signal(signal):
            continue

        name = str(signal.get("name") or "Finding")
        findings.append({
            "code": f"signal.{_slug(name)}",
            "title": name,
            "severity": normalize_severity(signal.get("severity")),
            "scope": "dataset",
            "status": signal.get("status"),
            "evidence": signal.get("evidence") or "",
            "recommendation": signal.get("recommendation") or "",
            "method": "signal_engine",
            "confidence": "deterministic",
        })

    return _sort_findings(findings)


def findings_from_target_intelligence(
    target_intelligence: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Normalize target-quality/leakage warnings into standard findings."""
    if not isinstance(target_intelligence, Mapping) or not target_intelligence.get("available"):
        return []

    target = str(target_intelligence.get("target_column") or "target")
    warnings = target_intelligence.get("warnings", [])
    if not isinstance(warnings, list):
        return []

    findings: list[dict[str, Any]] = []
    for warning in warnings:
        if not isinstance(warning, Mapping):
            continue
        code = str(warning.get("code") or "target.warning")
        evidence = str(warning.get("message") or "Target review is recommended.")

        if code == "target.id_like":
            title = "Identifier-like target"
            recommendation = (
                f"Verify that '{target}' represents an outcome rather than a record identifier "
                "before training a supervised model."
            )
        elif code == "target.high_missingness":
            title = "High target missingness"
            recommendation = (
                "Resolve or explicitly exclude rows with missing target labels before model training."
            )
        elif code == "target.high_cardinality_classification":
            title = "High-cardinality classification target"
            recommendation = (
                "Confirm that the target classes are intentional and have enough examples for reliable evaluation."
            )
        elif code.startswith("target.leakage."):
            feature = code.removeprefix("target.leakage.")
            title = f"Potential target leakage: {feature}"
            recommendation = (
                f"Review or remove '{feature}' before training if it contains information that would "
                "not be available at prediction time."
            )
        else:
            title = "Target review"
            recommendation = "Review the selected target and its relationship to the input features."

        findings.append({
            "code": code,
            "title": title,
            "severity": normalize_severity(warning.get("severity")),
            "scope": "target",
            "status": "Review",
            "evidence": evidence,
            "recommendation": recommendation,
            "method": "target_intelligence",
            "confidence": "deterministic",
        })

    return _sort_findings(findings)


def _quality_finding(
    *,
    code: str,
    title: str,
    severity: Any,
    evidence: str,
    recommendation: str,
    scope: str = "column",
) -> dict[str, Any]:
    return {
        "code": code,
        "title": title,
        "severity": normalize_severity(severity),
        "scope": scope,
        "status": "Review",
        "evidence": evidence,
        "recommendation": recommendation,
        "method": "quality_diagnostics",
        "confidence": "deterministic",
    }


def findings_from_quality_diagnostics(
    diagnostics: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Translate deterministic quality diagnostics into actionable findings."""
    if not isinstance(diagnostics, Mapping) or not diagnostics.get("available"):
        return []

    findings: list[dict[str, Any]] = []

    for item in diagnostics.get("identifier_duplicates", []):
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "identifier")
        duplicate_rows = int(item.get("duplicate_rows") or 0)
        findings.append(_quality_finding(
            code=f"quality.identifier_duplicates.{_slug(column)}",
            title=f"Duplicate identifiers: {column}",
            severity=item.get("severity", "high"),
            evidence=f"{duplicate_rows} rows share duplicated values in identifier-like column '{column}'.",
            recommendation=(
                f"Verify uniqueness rules for '{column}' and resolve duplicated identifiers before joins, "
                "deduplication, or model evaluation."
            ),
        ))

    for item in diagnostics.get("duplicate_columns", []):
        if not isinstance(item, Mapping):
            continue
        canonical = str(item.get("canonical_column") or "column")
        duplicates = [str(value) for value in item.get("duplicate_columns", [])]
        if not duplicates:
            continue
        findings.append(_quality_finding(
            code=f"quality.duplicate_columns.{_slug(canonical)}",
            title=f"Duplicate columns: {canonical}",
            severity=item.get("severity", "medium"),
            evidence=(
                f"'{canonical}' is exactly duplicated by: {', '.join(duplicates)}."
            ),
            recommendation=(
                "Remove or explicitly document redundant columns to reduce ambiguity, memory usage, and leakage risk."
            ),
        ))

    for item in diagnostics.get("quasi_constant_columns", []):
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "column")
        ratio = float(item.get("top_value_ratio") or 0)
        findings.append(_quality_finding(
            code=f"quality.quasi_constant.{_slug(column)}",
            title=f"Quasi-constant column: {column}",
            severity=item.get("severity", "low"),
            evidence=f"One value represents approximately {ratio:.1%} of non-missing sampled values.",
            recommendation=(
                f"Review whether '{column}' carries useful signal; near-constant features often add little analytical value."
            ),
        ))

    for item in diagnostics.get("coercion_candidates", []):
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "column")
        suggested = str(item.get("suggested_type") or "structured")
        ratio = float(item.get("parse_ratio") or 0)
        findings.append(_quality_finding(
            code=f"quality.coercion.{_slug(column)}",
            title=f"Type coercion candidate: {column}",
            severity=item.get("severity", "low"),
            evidence=f"Approximately {ratio:.1%} of sampled values parse cleanly as {suggested}.",
            recommendation=(
                f"Consider converting '{column}' to {suggested} after reviewing non-parsing values and preserving intended semantics."
            ),
        ))

    for item in diagnostics.get("category_normalisation", []):
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "column")
        groups = int(item.get("variant_group_count") or 0)
        findings.append(_quality_finding(
            code=f"quality.category_normalisation.{_slug(column)}",
            title=f"Category normalization issue: {column}",
            severity=item.get("severity", "medium"),
            evidence=f"Detected {groups} category groups that differ only by case or surrounding whitespace.",
            recommendation=(
                f"Normalize whitespace/casing in '{column}' with an explicit mapping before grouping, validation, or modelling."
            ),
        ))

    for item in diagnostics.get("blank_strings", []):
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "column")
        count = int(item.get("blank_count_in_sample") or 0)
        findings.append(_quality_finding(
            code=f"quality.blank_strings.{_slug(column)}",
            title=f"Blank strings: {column}",
            severity=item.get("severity", "medium"),
            evidence=f"Found {count} blank/whitespace-only values in the diagnostic sample.",
            recommendation=(
                f"Treat blank strings in '{column}' consistently as missing values or a documented category."
            ),
        ))

    for item in diagnostics.get("infinite_values", []):
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "column")
        count = int(item.get("infinite_count_in_sample") or 0)
        findings.append(_quality_finding(
            code=f"quality.infinite_values.{_slug(column)}",
            title=f"Infinite numeric values: {column}",
            severity=item.get("severity", "high"),
            evidence=f"Found {count} positive/negative infinite values in the diagnostic sample.",
            recommendation=(
                f"Replace or explicitly handle infinities in '{column}' before statistics, scaling, or model training."
            ),
        ))

    for item in diagnostics.get("mixed_object_types", []):
        if not isinstance(item, Mapping):
            continue
        column = str(item.get("column") or "column")
        types = item.get("python_types", {})
        findings.append(_quality_finding(
            code=f"quality.mixed_object_types.{_slug(column)}",
            title=f"Mixed Python types: {column}",
            severity=item.get("severity", "medium"),
            evidence=f"Object column contains multiple runtime value types: {types}.",
            recommendation=(
                f"Standardize the representation of '{column}' before serialization, joins, validation, or type conversion."
            ),
        ))

    for item in diagnostics.get("missingness_relationships", []):
        if not isinstance(item, Mapping):
            continue
        columns = [str(value) for value in item.get("columns", [])]
        if len(columns) != 2:
            continue
        jaccard = float(item.get("jaccard") or 0)
        findings.append(_quality_finding(
            code=f"quality.missingness_relationship.{_slug(columns[0])}.{_slug(columns[1])}",
            title=f"Linked missingness: {columns[0]} + {columns[1]}",
            severity=item.get("severity", "low"),
            evidence=f"Their missing-value masks have Jaccard similarity {jaccard:.2f} in the diagnostic sample.",
            recommendation=(
                "Investigate whether these columns are jointly missing because of one upstream process, segment, or collection rule."
            ),
            scope="dataset",
        ))

    return _sort_findings(findings)


def merge_findings(*groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge finding groups, de-duplicating by stable code."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for finding in group:
            code = str(finding.get("code") or "")
            if code and code in seen:
                continue
            if code:
                seen.add(code)
            merged.append(dict(finding))
    return _sort_findings(merged)


def recommendations_from_findings(
    findings: Iterable[dict[str, Any]],
) -> list[str]:
    """Return de-duplicated actionable recommendations in finding order."""
    recommendations: list[str] = []
    seen: set[str] = set()

    for finding in findings:
        recommendation = str(finding.get("recommendation") or "").strip()
        if not recommendation or recommendation in seen:
            continue
        seen.add(recommendation)
        recommendations.append(recommendation)

    return recommendations
