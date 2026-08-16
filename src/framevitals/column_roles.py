"""
Column Role Inference Engine
============================
Assigns semantic roles to each column based on name keywords, dtype, unique
ratio, missingness, statistical properties, and bounded value-pattern samples.
"""

import pandas as pd

from framevitals.semantic_types import infer_semantic_types

ID_KEYWORDS = [
    "id", "uuid", "hash", "key", "identifier", "index",
    "roll", "roll_number", "rollno", "roll number",
    "user_id", "userid", "student_id", "studentid",
    "application", "registration", "serial", "ticket",
    "account", "customer_id", "order_id", "transaction_id",
]

TIME_KEYWORDS = [
    "time", "date", "timestamp", "created", "updated",
    "event", "recv", "session", "datetime", "ts",
    "year", "month", "day", "hour", "minute", "second",
    "start_date", "end_date", "start_time", "end_time",
    "birth", "dob", "expiry",
]

PRICE_KEYWORDS = [
    "price", "cost", "amount", "fee", "charge", "revenue",
    "bid", "ask", "mark", "open", "close", "high", "low",
    "salary", "income", "wage", "rate", "premium", "discount",
]

VOLUME_KEYWORDS = [
    "volume", "qty", "quantity", "size", "count",
    "units", "shares", "lots", "items",
]

SENSITIVE_KEYWORDS = [
    "email", "phone", "name", "gender", "sex", "age",
    "race", "religion", "caste", "ethnicity",
    "salary", "income", "address", "location", "ssn",
    "passport", "license", "national",
]

SEQUENCE_KEYWORDS = [
    "sequence", "seq", "order", "rank", "position", "step",
]

TARGET_HINT_KEYWORDS = [
    "target", "label", "class", "output", "result",
    "outcome", "prediction", "status", "category", "grade",
    "pass", "fail", "approved", "rejected",
]

SEMANTIC_ROLE_MAP = {
    "email": "email_like",
    "url": "url_like",
    "uuid": "uuid_like",
    "ip_address": "ip_address_like",
    "phone": "phone_like",
    "percentage": "percentage_like",
    "currency": "currency_like",
    "json": "json_like",
    "boolean_token": "boolean_token_like",
}


def _name_matches(column_name: str, keywords: list) -> bool:
    lower = column_name.lower().replace("-", "_")
    return any(kw in lower for kw in keywords)


def _classify_missingness(missing_percent: float) -> str:
    if missing_percent >= 90:
        return "severe_missing"
    if missing_percent >= 50:
        return "very_high_missing"
    if missing_percent >= 20:
        return "high_missing"
    if missing_percent >= 5:
        return "moderate_missing"
    if missing_percent > 0:
        return "low_missing"
    return "complete"


def _infer_single_column_roles(column: str, series: pd.Series, rows: int) -> dict:
    roles = set()

    missing_pct = round(float(series.isna().mean() * 100), 2)
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / max(rows, 1)
    non_missing = int(series.notna().sum())

    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_bool = pd.api.types.is_bool_dtype(series)
    is_text = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    semantic = (
        infer_semantic_types(series)
        if is_text
        else {"primary": None, "candidates": [], "sample_size": 0}
    )
    semantic_primary = semantic.get("primary")

    if is_numeric:
        roles.add("numeric")
    if is_bool:
        roles.add("boolean")
    if is_text:
        roles.add("categorical")

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

    if _name_matches(column, ID_KEYWORDS) or (unique_ratio > 0.95 and not is_numeric):
        roles.add("id_like")
    if is_numeric and unique_ratio > 0.99 and _name_matches(column, ID_KEYWORDS):
        roles.add("id_like")
    if _name_matches(column, TIME_KEYWORDS):
        roles.add("time_like")
    if _name_matches(column, PRICE_KEYWORDS):
        roles.add("price_like")
    if _name_matches(column, VOLUME_KEYWORDS):
        roles.add("volume_like")
    if _name_matches(column, SENSITIVE_KEYWORDS):
        roles.add("sensitive")
    if _name_matches(column, SEQUENCE_KEYWORDS):
        roles.add("sequence_like")
    if _name_matches(column, TARGET_HINT_KEYWORDS):
        roles.add("target_hint")

    if semantic_primary in SEMANTIC_ROLE_MAP:
        roles.add(SEMANTIC_ROLE_MAP[semantic_primary])
    if semantic_primary in {"email", "phone", "ip_address"}:
        roles.add("sensitive")
    if semantic_primary == "uuid":
        roles.add("id_like")
    if semantic_primary == "currency":
        roles.add("price_like")

    if not is_numeric and not is_bool:
        sample = series.dropna().astype(str).head(30)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.7:
                roles.add("time_like")

    if is_text:
        lengths = series.dropna().astype(str).str.len()
        if len(lengths) > 0:
            avg_len = float(lengths.mean())
            max_len = int(lengths.max())
            if avg_len > 50 or max_len > 200:
                roles.add("long_text")
            else:
                roles.add("short_text")

    roles.add(_classify_missingness(missing_pct))

    excluded = {"id_like", "time_like", "sequence_like", "constant"}
    if not roles.intersection(excluded):
        roles.add("analysis_candidate")

    if "id_like" not in roles and "constant" not in roles:
        if "binary" in roles or "low_cardinality" in roles:
            roles.add("target_candidate")
        elif is_numeric and unique_count > 10:
            roles.add("regression_target_candidate")

    return {
        "roles": sorted(roles),
        "dtype": str(series.dtype),
        "missing_percent": missing_pct,
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "non_missing_count": non_missing,
        "is_numeric": bool(is_numeric),
        "is_categorical": bool(is_text or is_bool),
        "semantic_type": semantic_primary,
        "semantic_candidates": semantic.get("candidates", []),
        "semantic_sample_size": int(semantic.get("sample_size", 0)),
    }


def infer_column_roles(df: pd.DataFrame) -> dict:
    rows = len(df)
    return {
        col: _infer_single_column_roles(col, df[col], rows)
        for col in df.columns
    }


def get_columns_with_role(column_roles: dict, role: str) -> list:
    return [
        col for col, info in column_roles.items()
        if role in info["roles"]
    ]


def get_meaningful_numeric_columns(df: pd.DataFrame, column_roles: dict) -> list:
    candidates = []
    for col, info in column_roles.items():
        if not info["is_numeric"]:
            continue
        role_set = set(info["roles"])
        if role_set.intersection({"id_like", "time_like", "sequence_like", "constant"}):
            continue
        if info["non_missing_count"] < 10:
            continue
        candidates.append(col)
    return candidates


def get_meaningful_categorical_columns(df: pd.DataFrame, column_roles: dict) -> list:
    candidates = []
    for col, info in column_roles.items():
        if not info["is_categorical"]:
            continue
        role_set = set(info["roles"])
        if role_set.intersection({"id_like", "time_like", "constant"}):
            continue
        if info["unique_count"] < 2 or info["unique_count"] > 20:
            continue
        candidates.append(col)
    return candidates


def summarize_roles(column_roles: dict) -> dict:
    return {
        "total_columns": len(column_roles),
        "numeric_count": sum(1 for c in column_roles.values() if c["is_numeric"]),
        "categorical_count": sum(1 for c in column_roles.values() if c["is_categorical"]),
        "id_like": get_columns_with_role(column_roles, "id_like"),
        "time_like": get_columns_with_role(column_roles, "time_like"),
        "price_like": get_columns_with_role(column_roles, "price_like"),
        "volume_like": get_columns_with_role(column_roles, "volume_like"),
        "constant": get_columns_with_role(column_roles, "constant"),
        "high_cardinality": get_columns_with_role(column_roles, "high_cardinality"),
        "unique_like": get_columns_with_role(column_roles, "unique_like"),
        "binary": get_columns_with_role(column_roles, "binary"),
        "low_cardinality": get_columns_with_role(column_roles, "low_cardinality"),
        "target_candidates": get_columns_with_role(column_roles, "target_candidate"),
        "regression_target_candidates": get_columns_with_role(
            column_roles, "regression_target_candidate"
        ),
        "sensitive": get_columns_with_role(column_roles, "sensitive"),
        "email_like": get_columns_with_role(column_roles, "email_like"),
        "url_like": get_columns_with_role(column_roles, "url_like"),
        "uuid_like": get_columns_with_role(column_roles, "uuid_like"),
        "ip_address_like": get_columns_with_role(column_roles, "ip_address_like"),
        "phone_like": get_columns_with_role(column_roles, "phone_like"),
        "percentage_like": get_columns_with_role(column_roles, "percentage_like"),
        "currency_like": get_columns_with_role(column_roles, "currency_like"),
        "json_like": get_columns_with_role(column_roles, "json_like"),
        "boolean_token_like": get_columns_with_role(column_roles, "boolean_token_like"),
        "high_missing": [
            col for col, info in column_roles.items()
            if info["missing_percent"] >= 20
        ],
    }
