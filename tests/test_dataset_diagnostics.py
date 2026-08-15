import pandas as pd

from framevitals.advanced_indicators import (
    calculate_advanced_indicators,
)
from framevitals.column_roles import (
    infer_column_roles,
    summarize_roles,
)
from framevitals.dataset_signals import (
    detect_dataset_signals,
)
from framevitals.profiler import build_profile


def make_dataset():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [20, 25, 30, 35, 40],
        "income": [
            30000,
            35000,
            40000,
            45000,
            50000,
        ],
        "city": [
            "Pune",
            "Mumbai",
            "Pune",
            "Nashik",
            "Mumbai",
        ],
    })


def test_column_role_detection():
    df = make_dataset()

    roles = infer_column_roles(df)

    assert "customer_id" in roles
    assert "id_like" in roles["customer_id"]["roles"]


def test_role_summary():
    df = make_dataset()

    roles = infer_column_roles(df)
    summary = summarize_roles(roles)

    assert summary["total_columns"] == 4
    assert "customer_id" in summary["id_like"]


def test_dataset_signals():
    df = make_dataset()
    profile = build_profile(df)

    signals = detect_dataset_signals(
        df,
        profile,
    )

    assert signals["row_count"] == 5
    assert signals["column_count"] == 4
    assert signals["has_numeric_columns"] is True
    assert signals["has_id_like_columns"] is True


def test_advanced_indicators():
    df = make_dataset()

    result = calculate_advanced_indicators(df)

    assert "column_utility" in result
    assert "anomalies" in result
    assert "fairness" in result
    assert "freshness" in result
