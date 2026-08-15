import pandas as pd

from framevitals.feature_importance import (
    run_feature_importance,
)


def test_classification_feature_importance():
    rows = 80

    df = pd.DataFrame({
        "customer_id": [
            f"C{i}"
            for i in range(rows)
        ],
        "signal": [
            i % 2
            for i in range(rows)
        ],
        "noise": [
            (i * 7) % 13
            for i in range(rows)
        ],
        "city": [
            ["Pune", "Mumbai", "Nashik"][
                i % 3
            ]
            for i in range(rows)
        ],
        "target": [
            i % 2
            for i in range(rows)
        ],
    })

    result = run_feature_importance(
        df,
        target_column="target",
        task_type="classification",
    )

    assert result["available"] is True

    assert result[
        "global_importance"
    ]

    assert (
        result["global_importance"]
        == result["top_features"]
    )

    features = {
        item["feature"]
        for item in result[
            "global_importance"
        ]
    }

    assert "signal" in features

    assert "customer_id" in result[
        "dropped_columns"
    ]

    assert result[
        "mutual_information"
    ]


def test_regression_feature_importance():
    rows = 80

    df = pd.DataFrame({
        "x": list(range(rows)),
        "noise": [
            (i * 11) % 17
            for i in range(rows)
        ],
        "target": [
            (i * 4) + (i % 3)
            for i in range(rows)
        ],
    })

    result = run_feature_importance(
        df,
        target_column="target",
        task_type="regression",
    )

    assert result["available"] is True
    assert result["task_type"] == "regression"

    features = {
        item["feature"]
        for item in result[
            "global_importance"
        ]
    }

    assert "x" in features


def test_categorical_name_with_underscore():
    rows = 60

    df = pd.DataFrame({
        "customer_segment": [
            "premium"
            if i % 2 == 0
            else "standard"
            for i in range(rows)
        ],
        "target": [
            i % 2
            for i in range(rows)
        ],
    })

    result = run_feature_importance(
        df,
        target_column="target",
        task_type="classification",
    )

    assert result["available"] is True

    features = {
        item["feature"]
        for item in result[
            "global_importance"
        ]
    }

    assert "customer_segment" in features
    assert "customer" not in features


def test_feature_importance_invalid_target():
    df = pd.DataFrame({
        "x": list(range(30)),
    })

    result = run_feature_importance(
        df,
        target_column="missing",
        task_type="regression",
    )

    assert result["available"] is False
