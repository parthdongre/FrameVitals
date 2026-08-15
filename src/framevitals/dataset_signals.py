"""
Dataset Signal Detector
========================
Produces a flat dictionary of boolean + numeric signals describing
the structural characteristics of the dataset.

These signals drive the analysis selector engine — they answer
questions like "does this dataset have missing values?" or
"are there ID-like columns?" without hardcoding domain logic.
"""

import re

import pandas as pd

from framevitals.column_roles import (
    get_columns_with_role,
    get_meaningful_numeric_columns,
    infer_column_roles,
)


_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TEXT_DTYPES = ["object", "string", "category"]


def _detect_long_text_columns(df: pd.DataFrame) -> list:
    result = []
    for col in df.select_dtypes(include=_TEXT_DTYPES).columns:
        lengths = df[col].dropna().astype(str).str.len()
        if len(lengths) == 0:
            continue
        if float(lengths.mean()) > 50 or int(lengths.max()) > 200:
            result.append(col)
    return result


def _detect_email_columns(df: pd.DataFrame) -> list:
    result = []
    for col in df.select_dtypes(include=_TEXT_DTYPES).columns:
        sample = df[col].dropna().astype(str).head(100)
        if sample.empty:
            continue
        match_ratio = sample.apply(lambda v: bool(_EMAIL_PATTERN.match(v))).mean()
        if match_ratio >= 0.5:
            result.append(col)
    return result


def _detect_potential_leakage(df: pd.DataFrame, column_roles: dict) -> bool:
    """Return whether a very high non-ID numeric correlation suggests leakage."""
    meaningful = get_meaningful_numeric_columns(df, column_roles)
    if len(meaningful) < 2:
        return False
    try:
        corr = df[meaningful].corr(numeric_only=True).abs()
        for i, a in enumerate(meaningful):
            for b in meaningful[i + 1 :]:
                val = corr.loc[a, b]
                if pd.notna(val) and val >= 0.98:
                    return True
    except Exception:
        pass
    return False


def detect_dataset_signals(
    df: pd.DataFrame,
    profile: dict,
    column_roles: dict | None = None,
) -> dict:
    """Produce structural signals, reusing cached pipeline metadata when supplied.

    ``column_roles`` is optional for backward compatibility. The main analysis
    pipeline passes its already-computed role map so this stage does not repeat
    the most expensive per-column semantic scan.
    """
    rows, cols = df.shape

    numeric_cols = profile.get("numeric_columns", [])
    categorical_cols = profile.get("categorical_columns", [])
    date_cols = profile.get("date_columns", [])

    missing_cells = sum(
        int(value)
        for value in profile.get("missing_counts", {}).values()
        if value is not None
    )
    total_cells = max(rows * cols, 1)
    missing_pct = round(missing_cells / total_cells * 100, 2)

    cached_duplicates = profile.get("duplicate_rows")
    duplicate_rows = int(
        cached_duplicates if cached_duplicates is not None else df.duplicated().sum()
    )

    if column_roles is None:
        column_roles = infer_column_roles(df)

    id_like = get_columns_with_role(column_roles, "id_like")
    price_like = get_columns_with_role(column_roles, "price_like")
    volume_like = get_columns_with_role(column_roles, "volume_like")
    time_like = get_columns_with_role(column_roles, "time_like")
    sensitive = get_columns_with_role(column_roles, "sensitive")
    constant = get_columns_with_role(column_roles, "constant")
    binary = get_columns_with_role(column_roles, "binary")
    low_card = get_columns_with_role(column_roles, "low_cardinality")
    high_card = get_columns_with_role(column_roles, "high_cardinality")
    unique_like = get_columns_with_role(column_roles, "unique_like")
    target_candidates = get_columns_with_role(column_roles, "target_candidate")

    # Column-role inference already computes the same full-column text-length
    # rule, so reuse it instead of scanning text columns a second time.
    long_text = get_columns_with_role(column_roles, "long_text")
    email_like = _detect_email_columns(df)
    has_leakage_risk = _detect_potential_leakage(df, column_roles)

    lower_map = {c.lower(): c for c in df.columns}
    has_bid_ask = "bid" in lower_map and "ask" in lower_map

    has_time_series = len(time_like) > 0 and len(numeric_cols) >= 1 and rows >= 20

    return {
        "row_count": rows,
        "column_count": cols,
        "numeric_column_count": len(numeric_cols),
        "categorical_column_count": len(categorical_cols),
        "date_column_count": len(date_cols),
        "has_numeric_columns": len(numeric_cols) > 0,
        "has_multiple_numeric_columns": len(numeric_cols) >= 2,
        "has_categorical_columns": len(categorical_cols) > 0,
        "has_datetime_columns": len(date_cols) > 0,
        "has_text_columns": len(categorical_cols) > 0,
        "has_long_text_columns": len(long_text) > 0,
        "has_missing_values": missing_cells > 0,
        "has_high_missingness": missing_pct >= 20,
        "has_duplicates": duplicate_rows > 0,
        "has_id_columns": len(id_like) > 0,
        "has_id_like_columns": len(id_like) > 0,
        "has_high_cardinality_columns": len(high_card) > 0,
        "has_binary_columns": len(binary) > 0,
        "has_low_cardinality_columns": len(low_card) > 0,
        "has_constant_columns": len(constant) > 0,
        "has_unique_like_columns": len(unique_like) > 0,
        "has_price_like_columns": len(price_like) > 0,
        "has_bid_ask_columns": has_bid_ask,
        "has_volume_like_columns": len(volume_like) > 0,
        "has_timestamp_like_columns": len(time_like) > 0,
        "has_time_series_structure": has_time_series,
        "has_sensitive_column_candidates": len(sensitive) > 0,
        "has_email_like_columns": len(email_like) > 0,
        "has_potential_leakage": has_leakage_risk,
        "has_target_candidates": len(target_candidates) > 0,
        "is_small_dataset": rows < 200,
        "is_large_dataset": rows >= 100000,
        "is_wide_dataset": cols >= 50,
        "missing_percent": missing_pct,
        "duplicate_rows": duplicate_rows,
        "id_like_columns": id_like,
        "price_like_columns": price_like,
        "volume_like_columns": volume_like,
        "timestamp_like_columns": time_like,
        "sensitive_columns": sensitive,
        "constant_columns": constant,
        "email_like_columns": email_like,
        "long_text_columns": long_text,
        "binary_columns": binary,
        "low_cardinality_columns": low_card,
        "high_cardinality_columns": high_card,
        "unique_like_columns": unique_like,
        "target_candidate_columns": target_candidates,
        "column_roles": column_roles,
    }
