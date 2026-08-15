import pandas as pd

from framevitals.deep_statistics_v2 import (
    run_deep_statistics_v2,
)
from framevitals.text_profile import (
    profile_text_columns,
)
from framevitals.time_series import (
    detect_and_analyze_time_series,
)


def test_deep_statistics():
    df = pd.DataFrame({
        "age": [
            20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31,
        ],
        "income": [
            30000, 32000, 34000, 36000,
            38000, 40000, 42000, 44000,
            46000, 48000, 50000, 52000,
        ],
        "city": [
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
        ],
    })

    result = run_deep_statistics_v2(df)

    assert result["version"] == "v2"

    assert "age" in result["numeric_columns"]
    assert "income" in result["numeric_columns"]
    assert "city" in result["categorical_columns"]

    assert result["summary"]["numeric_count"] == 2

def test_text_profile():
    df = pd.DataFrame({
        "review": [
            "This product works very well and feels great.",
            "The application is simple and very useful.",
            "I really like this dataset analysis tool.",
            "The interface works well for data analysis.",
            "This tool provides helpful dataset information.",
            "The report contains useful statistical insights.",
            "The analysis explains the dataset very clearly.",
            "This product has good features for analysts.",
            "The software provides detailed data diagnostics.",
            "This dataset contains several useful observations.",
        ],
    })

    result = profile_text_columns(df)

    assert result["available"] is True
    assert "review" in result["detected_columns"]
    assert "review" in result["profiles"]


def test_time_series_detection():
    df = pd.DataFrame({
        "date": pd.date_range(
            "2026-01-01",
            periods=40,
            freq="D",
        ),
        "sales": [
            100 + i
            for i in range(40)
        ],
    })

    result = detect_and_analyze_time_series(
        df,
        target_column="sales",
    )

    assert result["available"] is True
    assert result["detected_date_column"] == "date"
    assert result["numeric_column"] == "sales"
    assert result["n_observations"] == 40
