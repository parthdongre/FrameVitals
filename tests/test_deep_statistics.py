import pandas as pd

from framevitals.deep_statistics import (
    run_deep_statistics,
)


def make_dataset():
    return pd.DataFrame({
        "x": list(range(1, 31)),
        "y": [
            value * 2
            for value in range(1, 31)
        ],
        "group": [
            "A",
            "B",
        ] * 15,
        "region": [
            "North",
            "South",
        ] * 15,
    })


def test_numeric_statistics():
    result = run_deep_statistics(
        make_dataset()
    )

    numeric = result[
        "numeric_statistics"
    ]

    assert "x" in numeric
    assert "y" in numeric

    assert numeric["x"]["count"] == 30
    assert numeric["x"]["min"] == 1
    assert numeric["x"]["max"] == 30

    assert numeric["x"][
        "skewness_label"
    ] == "Approximately Symmetric"


def test_correlation_insights():
    result = run_deep_statistics(
        make_dataset()
    )

    correlation = result[
        "correlation_insights"
    ]

    assert correlation["available"] is True

    pair = next(
        item
        for item in correlation["top_pairs"]
        if {
            item["column_a"],
            item["column_b"],
        } == {"x", "y"}
    )

    assert pair["correlation"] == 1.0
    assert pair["strength"] == "Very Strong"


def test_categorical_statistics():
    result = run_deep_statistics(
        make_dataset()
    )

    categorical = result[
        "categorical_statistics"
    ]

    assert "group" in categorical

    assert categorical["group"][
        "unique_count"
    ] == 2

    assert categorical["group"][
        "cardinality_label"
    ] == "Low Cardinality"


def test_chi_square_relationships():
    result = run_deep_statistics(
        make_dataset()
    )

    relationships = result[
        "chi_square_relationships"
    ]

    assert len(relationships) >= 1

    pair = next(
        item
        for item in relationships
        if {
            item["column_a"],
            item["column_b"],
        } == {"group", "region"}
    )

    assert pair[
        "relationship_likely"
    ] is True
