"""
Dataset Signal Detector
========================
Produces a flat dictionary of boolean + numeric signals describing the
structural characteristics of the dataset.

Signals reuse the role map and profile when available so semantic/type scans
and numeric correlation work are performed once per pipeline run.
"""

from framevitals.column_roles import (
    get_columns_with_role,
    infer_column_roles,
)


def _detect_potential_leakage(profile: dict, column_roles: dict) -> tuple[bool, bool]:
    """Inspect cached profiler correlations for very high non-ID relationships."""
    correlations = profile.get("correlations", {}) or {}
    correlation_metadata = profile.get("correlation_metadata", {}) or {}
    truncated = bool(correlation_metadata.get("truncated", False))

    excluded = {"id_like", "time_like", "sequence_like", "constant"}
    meaningful = {
        column
        for column, info in column_roles.items()
        if info.get("is_numeric") and not set(info.get("roles", [])).intersection(excluded)
    }

    for left, values in correlations.items():
        if left not in meaningful or not isinstance(values, dict):
            continue
        for right, value in values.items():
            if right == left or right not in meaningful or value is None:
                continue
            try:
                if abs(float(value)) >= 0.98:
                    return True, truncated
            except (TypeError, ValueError):
                continue
    return False, truncated


def detect_dataset_signals(
    df,
    profile: dict,
    column_roles: dict | None = None,
) -> dict:
    """Produce structural signals, reusing cached pipeline metadata when supplied.

    ``column_roles`` is optional for backward compatibility. The main analysis
    pipeline passes its already-computed role map so this stage does not repeat
    per-column semantic scans.
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
    long_text = get_columns_with_role(column_roles, "long_text")

    # Semantic/value-pattern roles are already computed during role inference.
    email_like = get_columns_with_role(column_roles, "email_like")
    url_like = get_columns_with_role(column_roles, "url_like")
    uuid_like = get_columns_with_role(column_roles, "uuid_like")
    ip_address_like = get_columns_with_role(column_roles, "ip_address_like")
    phone_like = get_columns_with_role(column_roles, "phone_like")
    percentage_like = get_columns_with_role(column_roles, "percentage_like")
    currency_like = get_columns_with_role(column_roles, "currency_like")
    json_like = get_columns_with_role(column_roles, "json_like")
    boolean_token_like = get_columns_with_role(column_roles, "boolean_token_like")

    has_leakage_risk, leakage_scan_truncated = _detect_potential_leakage(
        profile,
        column_roles,
    )

    lower_map = {str(c).lower(): c for c in df.columns}
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
        "has_url_like_columns": len(url_like) > 0,
        "has_uuid_like_columns": len(uuid_like) > 0,
        "has_ip_address_like_columns": len(ip_address_like) > 0,
        "has_phone_like_columns": len(phone_like) > 0,
        "has_percentage_like_columns": len(percentage_like) > 0,
        "has_currency_like_columns": len(currency_like) > 0,
        "has_json_like_columns": len(json_like) > 0,
        "has_boolean_token_like_columns": len(boolean_token_like) > 0,
        "has_potential_leakage": has_leakage_risk,
        "leakage_scan_truncated": leakage_scan_truncated,
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
        "url_like_columns": url_like,
        "uuid_like_columns": uuid_like,
        "ip_address_like_columns": ip_address_like,
        "phone_like_columns": phone_like,
        "percentage_like_columns": percentage_like,
        "currency_like_columns": currency_like,
        "json_like_columns": json_like,
        "boolean_token_like_columns": boolean_token_like,
        "long_text_columns": long_text,
        "binary_columns": binary,
        "low_cardinality_columns": low_card,
        "high_cardinality_columns": high_card,
        "unique_like_columns": unique_like,
        "target_candidate_columns": target_candidates,
        "column_roles": column_roles,
    }
