import pandas as pd

from framevitals.segment_analysis import (
    run_segment_analysis,
)
from framevitals.target_analyzer import (
    analyze_target,
)


def test_classification_target_analysis():
    df = pd.DataFrame({
        "target": [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 1, 1,
        ],
    })

    result = analyze_target(
        df,
        target_column="target",
    )

    assert result["available"] is True
    assert result["task_type"] == (
        "classification"
    )

    details = result["details"]

    assert details["class_count"] == 2
    assert details["majority_ratio"] == 90.0
    assert details["imbalance_status"] == (
        "Severely Imbalanced"
    )

    assert len(result["warnings"]) >= 1


def test_regression_target_analysis():
    df = pd.DataFrame({
        "target": list(
            range(1, 31)
        ),
    })

    result = analyze_target(
        df,
        target_column="target",
    )

    assert result["available"] is True
    assert result["task_type"] == "regression"

    details = result["details"]

    assert details["count"] == 30
    assert details["min"] == 1.0
    assert details["max"] == 30.0


def test_segment_analysis_with_numeric_target():
    df = pd.DataFrame({
        "city": [
            "Pune",
            "Mumbai",
        ] * 10,
        "target": [
            value
            for value in range(20)
        ],
    })

    result = run_segment_analysis(
        df,
        target_column="target",
    )

    assert result["available"] is True
    assert "city" in result[
        "segment_columns"
    ]

    city_result = next(
        item
        for item in result["results"]
        if item["segment_column"] == "city"
    )

    assert city_result["analysis_type"] == (
        "numeric_target_by_segment"
    )

    assert len(city_result["groups"]) == 2


def test_segment_distribution_without_target():
    df = pd.DataFrame({
        "department": [
            "IT",
            "CS",
            "ENTC",
            "IT",
            "CS",
            "IT",
        ],
        "score": [
            10, 20, 30, 40, 50, 60,
        ],
    })

    result = run_segment_analysis(df)

    assert result["available"] is True

    segment = result["results"][0]

    assert segment["analysis_type"] == (
        "segment_distribution"
    )
