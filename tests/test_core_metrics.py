import pandas as pd

from framevitals.health_score import calculate_health_score
from framevitals.ml_readiness import calculate_ml_readiness
from framevitals.profiler import build_profile


def make_dataset():
    return pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "income": [30000, 32000, 35000, 37000, 40000],
        "city": ["Pune", "Mumbai", "Pune", "Nashik", "Mumbai"],
    })


def test_profile_shape():
    df = make_dataset()

    profile = build_profile(df)

    assert profile["shape"]["rows"] == 5
    assert profile["shape"]["columns"] == 3

    assert "age" in profile["numeric_columns"]
    assert "income" in profile["numeric_columns"]


def test_profile_missing_values():
    df = make_dataset()
    df.loc[0, "income"] = None

    profile = build_profile(df)

    assert profile["missing_counts"]["income"] == 1


def test_health_score_range():
    df = make_dataset()
    profile = build_profile(df)

    health = calculate_health_score(
        df,
        profile,
    )

    assert 0 <= health["overall_score"] <= 100

    assert health["label"] in {
        "Excellent",
        "Good",
        "Moderate",
        "Poor",
        "Critical",
    }


def test_ml_readiness_range():
    df = make_dataset()

    readiness = calculate_ml_readiness(df)

    assert 0 <= readiness["score"] <= 100

    assert readiness["label"] in {
        "Ready",
        "Mostly Ready",
        "Partially Ready",
        "Not Ready",
    }


def test_ml_readiness_detects_categories():
    df = make_dataset()

    readiness = calculate_ml_readiness(df)

    assert "city" in readiness["categorical_columns"]
    assert readiness["issues"]["encoding_required_count"] >= 1
