"""Deterministic data-quality diagnostics for tabular datasets.

The diagnostics in this module are intentionally explainable and resource
bounded. Existing profile/role metadata is reused where possible. Potentially
expensive value-level checks operate on a deterministic bounded sample, while
exact full-column comparisons are reserved for duplicate-column candidates
identified by a sample fingerprint first.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_MAX_SAMPLE_ROWS = 5_000
DEFAULT_MAX_COLUMNS = 100
DEFAULT_MAX_MISSINGNESS_COLUMNS = 25


def _bounded_series(series: pd.Series, max_rows: int) -> tuple[pd.Series, bool]:
    if len(series) <= max_rows:
        return series, False
    positions = np.linspace(0, len(series) - 1, num=max_rows, dtype=int)
    positions = np.unique(positions)
    return series.iloc[positions], True


def _bounded_frame(frame: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    if len(frame) <= max_rows:
        return frame, False
    positions = np.linspace(0, len(frame) - 1, num=max_rows, dtype=int)
    positions = np.unique(positions)
    return frame.iloc[positions], True


def _text_like(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _column_info(column_roles: Mapping[str, Any], column: str) -> dict[str, Any]:
    value = column_roles.get(column, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _profile_mapping(profile: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = profile.get(key, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _primary_key_candidates(
    df: pd.DataFrame,
    column_roles: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = len(df)
    if rows < 2:
        return []

    candidates: list[dict[str, Any]] = []
    for column in df.columns:
        info = _column_info(column_roles, column)
        unique_count = int(info.get("unique_count") or 0)
        non_missing = int(info.get("non_missing_count") or 0)
        if unique_count != rows or non_missing != rows:
            continue

        roles = set(info.get("roles", []))
        confidence = "high" if "id_like" in roles else "medium"
        candidates.append({
            "column": column,
            "confidence": confidence,
            "reason": (
                "Column is complete, unique for every row, and identifier-like."
                if confidence == "high"
                else "Column is complete and unique for every row."
            ),
        })

    candidates.sort(key=lambda item: (item["confidence"] != "high", item["column"]))
    return candidates


def _identifier_duplicates(
    df: pd.DataFrame,
    column_roles: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for column in df.columns:
        info = _column_info(column_roles, column)
        if "id_like" not in set(info.get("roles", [])):
            continue
        clean = df[column].dropna()
        if clean.empty:
            continue
        duplicate_mask = clean.duplicated(keep=False)
        duplicate_rows = int(duplicate_mask.sum())
        if not duplicate_rows:
            continue
        duplicated_values = int(clean.loc[duplicate_mask].nunique(dropna=True))
        results.append({
            "column": column,
            "duplicate_rows": duplicate_rows,
            "duplicated_identifier_values": duplicated_values,
            "severity": "high",
        })
    return results


def _quasi_constants(
    df: pd.DataFrame,
    column_roles: Mapping[str, Any],
    *,
    max_rows: int,
    threshold: float = 0.95,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for column in df.columns:
        info = _column_info(column_roles, column)
        if int(info.get("unique_count") or 0) <= 1:
            continue
        sample, sampled = _bounded_series(df[column], max_rows)
        clean = sample.dropna()
        if len(clean) < 10:
            continue
        counts = clean.value_counts(dropna=True)
        if counts.empty:
            continue
        ratio = float(counts.iloc[0] / len(clean))
        if ratio < threshold:
            continue
        results.append({
            "column": column,
            "top_value": str(counts.index[0]),
            "top_value_ratio": round(ratio, 4),
            "sampled": sampled,
            "sample_rows": int(len(sample)),
            "severity": "medium" if ratio >= 0.99 else "low",
        })
    return results


def _series_fingerprint(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype="uint64")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(series.dtype).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _duplicate_columns(
    df: pd.DataFrame,
    *,
    max_rows: int,
    max_columns: int,
) -> list[dict[str, Any]]:
    columns = list(df.columns[:max_columns])
    sample, sampled = _bounded_frame(df[columns], max_rows)
    buckets: dict[str, list[str]] = {}

    for column in columns:
        fingerprint = _series_fingerprint(sample[column])
        buckets.setdefault(fingerprint, []).append(column)

    results: list[dict[str, Any]] = []
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        anchor = candidates[0]
        duplicates = [
            candidate
            for candidate in candidates[1:]
            if df[anchor].equals(df[candidate])
        ]
        if duplicates:
            results.append({
                "canonical_column": anchor,
                "duplicate_columns": duplicates,
                "sampled_for_fingerprint": sampled,
                "confirmed_with_full_equality": True,
                "severity": "medium",
            })
    return results


def _normalise_numeric_text(values: pd.Series) -> pd.Series:
    cleaned = values.str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.str.replace(r"^[\$€£₹]\s*", "", regex=True)
    cleaned = cleaned.str.replace(r"%$", "", regex=True)
    return cleaned


def _coercion_candidates(
    df: pd.DataFrame,
    column_roles: Mapping[str, Any],
    *,
    max_rows: int,
    threshold: float = 0.95,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        if not _text_like(series):
            continue

        info = _column_info(column_roles, column)
        roles = set(info.get("roles", []))
        sample, sampled = _bounded_series(series, max_rows)
        clean = sample.dropna().astype("string").str.strip()
        clean = clean[clean.ne("")]
        if len(clean) < 10:
            continue

        numeric_values = _normalise_numeric_text(clean)
        numeric_ratio = float(pd.to_numeric(numeric_values, errors="coerce").notna().mean())

        if numeric_ratio >= threshold and "id_like" not in roles:
            semantic = info.get("semantic_type")
            transform = "parse numeric strings"
            if semantic == "currency":
                transform = "remove currency symbols/separators, then parse numeric"
            elif semantic == "percentage":
                transform = "remove percent suffix, then parse numeric"
            results.append({
                "column": column,
                "suggested_type": "numeric",
                "parse_ratio": round(numeric_ratio, 4),
                "transform": transform,
                "sampled": sampled,
                "sample_rows": int(len(sample)),
                "severity": "low",
            })
            continue

        if numeric_ratio >= 0.50:
            # Avoid treating mostly-numeric identifiers as dates.
            continue

        parsed_dates = pd.to_datetime(clean, errors="coerce", format="mixed")
        date_ratio = float(parsed_dates.notna().mean())
        if date_ratio >= threshold:
            results.append({
                "column": column,
                "suggested_type": "datetime",
                "parse_ratio": round(date_ratio, 4),
                "transform": "parse datetime strings",
                "sampled": sampled,
                "sample_rows": int(len(sample)),
                "severity": "low",
            })

    return results


def _category_normalisation_issues(
    df: pd.DataFrame,
    column_roles: Mapping[str, Any],
    *,
    max_rows: int,
    max_unique: int = 200,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        if not _text_like(series):
            continue
        info = _column_info(column_roles, column)
        if "long_text" in set(info.get("roles", [])):
            continue

        sample, sampled = _bounded_series(series, max_rows)
        clean = sample.dropna().astype("string")
        unique_values = clean.unique().tolist()
        if len(unique_values) < 2 or len(unique_values) > max_unique:
            continue

        groups: dict[str, list[str]] = {}
        for raw in unique_values:
            raw_text = str(raw)
            canonical = raw_text.strip().casefold()
            groups.setdefault(canonical, []).append(raw_text)

        variants = [
            {"canonical": canonical, "variants": sorted(set(raw_values))}
            for canonical, raw_values in groups.items()
            if len(set(raw_values)) > 1
        ]
        if not variants:
            continue

        variants.sort(key=lambda item: (-len(item["variants"]), item["canonical"]))
        results.append({
            "column": column,
            "variant_group_count": len(variants),
            "groups": variants[:10],
            "sampled": sampled,
            "sample_rows": int(len(sample)),
            "severity": "medium",
        })

    return results


def _blank_string_issues(
    df: pd.DataFrame,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        if not _text_like(series):
            continue
        sample, sampled = _bounded_series(series, max_rows)
        non_null = sample.dropna().astype("string")
        if non_null.empty:
            continue
        blank_count = int(non_null.str.strip().eq("").sum())
        if not blank_count:
            continue
        results.append({
            "column": column,
            "blank_count_in_sample": blank_count,
            "blank_ratio_in_sample": round(float(blank_count / len(non_null)), 4),
            "sampled": sampled,
            "sample_rows": int(len(sample)),
            "severity": "medium",
        })
    return results


def _infinity_issues(
    df: pd.DataFrame,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        sample, sampled = _bounded_series(df[column], max_rows)
        values = pd.to_numeric(sample, errors="coerce").to_numpy(dtype="float64")
        count = int(np.isinf(values).sum())
        if not count:
            continue
        results.append({
            "column": column,
            "infinite_count_in_sample": count,
            "sampled": sampled,
            "sample_rows": int(len(sample)),
            "severity": "high",
        })
    return results


def _mixed_object_types(
    df: pd.DataFrame,
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_object_dtype(series):
            continue
        sample, sampled = _bounded_series(series, max_rows)
        clean = sample.dropna()
        if len(clean) < 2:
            continue
        counts = clean.map(lambda value: type(value).__name__).value_counts()
        if len(counts) <= 1:
            continue
        results.append({
            "column": column,
            "python_types": {str(key): int(value) for key, value in counts.items()},
            "sampled": sampled,
            "sample_rows": int(len(sample)),
            "severity": "medium",
        })
    return results


def _missingness_relationships(
    df: pd.DataFrame,
    profile: Mapping[str, Any],
    *,
    max_rows: int,
    max_columns: int,
    jaccard_threshold: float = 0.70,
) -> list[dict[str, Any]]:
    missing_counts = _profile_mapping(profile, "missing_counts")
    candidates = [
        column
        for column in df.columns
        if int(missing_counts.get(column) or 0) > 0
    ][:max_columns]

    if len(candidates) < 2:
        return []

    sample, sampled = _bounded_frame(df[candidates], max_rows)
    masks = {column: sample[column].isna().to_numpy() for column in candidates}
    relationships: list[dict[str, Any]] = []

    for left, right in combinations(candidates, 2):
        left_mask = masks[left]
        right_mask = masks[right]
        union = int(np.logical_or(left_mask, right_mask).sum())
        if not union:
            continue
        intersection = int(np.logical_and(left_mask, right_mask).sum())
        if intersection < 3:
            continue
        jaccard = intersection / union
        if jaccard < jaccard_threshold:
            continue
        relationships.append({
            "columns": [left, right],
            "co_missing_rows_in_sample": intersection,
            "jaccard": round(float(jaccard), 4),
            "sampled": sampled,
            "sample_rows": int(len(sample)),
            "severity": "medium" if jaccard >= 0.90 else "low",
        })

    relationships.sort(key=lambda item: (-item["jaccard"], item["columns"]))
    return relationships[:30]


def run_quality_diagnostics(
    df: pd.DataFrame,
    *,
    profile: Mapping[str, Any] | None = None,
    column_roles: Mapping[str, Any] | None = None,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    max_missingness_columns: int = DEFAULT_MAX_MISSINGNESS_COLUMNS,
) -> dict[str, Any]:
    """Return a bounded, deterministic set of practical data-quality checks."""
    if max_sample_rows < 10:
        raise ValueError("max_sample_rows must be at least 10.")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")
    if max_missingness_columns < 2:
        raise ValueError("max_missingness_columns must be at least 2.")

    if profile is None:
        from framevitals.profiler import build_profile

        profile = build_profile(df)
    if column_roles is None:
        from framevitals.column_roles import infer_column_roles

        column_roles = infer_column_roles(df)

    selected_columns = list(df.columns[:max_columns])
    work = df[selected_columns]
    selected_roles = {
        column: _column_info(column_roles, column)
        for column in selected_columns
    }

    primary_keys = _primary_key_candidates(work, selected_roles)
    identifier_duplicates = _identifier_duplicates(work, selected_roles)
    quasi_constants = _quasi_constants(
        work,
        selected_roles,
        max_rows=max_sample_rows,
    )
    duplicate_columns = _duplicate_columns(
        work,
        max_rows=max_sample_rows,
        max_columns=max_columns,
    )
    coercions = _coercion_candidates(
        work,
        selected_roles,
        max_rows=max_sample_rows,
    )
    category_normalisation = _category_normalisation_issues(
        work,
        selected_roles,
        max_rows=max_sample_rows,
    )
    blank_strings = _blank_string_issues(work, max_rows=max_sample_rows)
    infinities = _infinity_issues(work, max_rows=max_sample_rows)
    mixed_types = _mixed_object_types(work, max_rows=max_sample_rows)
    missingness_relationships = _missingness_relationships(
        work,
        profile,
        max_rows=max_sample_rows,
        max_columns=max_missingness_columns,
    )

    duplicate_rows = int(profile.get("duplicate_rows", 0) or 0)
    checks = {
        "primary_key_candidates": primary_keys,
        "identifier_duplicates": identifier_duplicates,
        "quasi_constant_columns": quasi_constants,
        "duplicate_columns": duplicate_columns,
        "coercion_candidates": coercions,
        "category_normalisation": category_normalisation,
        "blank_strings": blank_strings,
        "infinite_values": infinities,
        "mixed_object_types": mixed_types,
        "missingness_relationships": missingness_relationships,
    }

    issue_count = sum(
        len(value)
        for key, value in checks.items()
        if key != "primary_key_candidates"
    )

    return {
        "available": True,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "columns_checked": len(selected_columns),
        "truncated_columns": len(df.columns) > len(selected_columns),
        "max_sample_rows": int(max_sample_rows),
        "duplicate_rows": duplicate_rows,
        "summary": {
            "issue_groups": sum(bool(value) for key, value in checks.items() if key != "primary_key_candidates"),
            "issue_count": issue_count + (1 if duplicate_rows else 0),
            "primary_key_candidate_count": len(primary_keys),
        },
        **checks,
    }
