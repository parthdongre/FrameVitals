import pandas as pd

from framevitals.baseline_model import (
    run_baseline_model,
)


def test_classification_baseline():
    rows = 60

    df = pd.DataFrame({
        "customer_id": [
            f"C{i}"
            for i in range(rows)
        ],
        "age": [
            20 + (i % 30)
            for i in range(rows)
        ],
        "city": [
            "Pune" if i % 2 == 0
            else "Mumbai"
            for i in range(rows)
        ],
        "target": [
            i % 2
            for i in range(rows)
        ],
    })

    result = run_baseline_model(
        df,
        target_column="target",
        task_type="classification",
    )

    assert result["available"] is True

    assert result["task_type"] == (
        "classification"
    )

    assert result["model"] == (
        "RandomForestClassifier"
    )

    assert result["baseline_model"] == (
        "DummyClassifier"
    )

    assert result["primary_metric"] == (
        "f1_weighted"
    )

    assert (
        result["train_rows"]
        + result["test_rows"]
        == rows
    )

    assert "customer_id" in result[
        "dropped_columns"
    ]

    assert "age" in result[
        "used_numeric_features"
    ]

    assert "city" in result[
        "used_categorical_features"
    ]


def test_regression_baseline():
    rows = 60

    df = pd.DataFrame({
        "feature": list(range(rows)),
        "category": [
            "A" if i % 2 == 0
            else "B"
            for i in range(rows)
        ],
        "target": [
            (i * 3) + (i % 4)
            for i in range(rows)
        ],
    })

    result = run_baseline_model(
        df,
        target_column="target",
        task_type="regression",
    )

    assert result["available"] is True

    assert result["task_type"] == (
        "regression"
    )

    assert result["model"] == (
        "RandomForestRegressor"
    )

    assert result["baseline_model"] == (
        "DummyRegressor"
    )

    assert result["primary_metric"] == "r2"

    assert "r2" in result[
        "model_metrics"
    ]

    assert "rmse" in result[
        "model_metrics"
    ]


def test_baseline_invalid_target():
    df = pd.DataFrame({
        "x": list(range(40)),
    })

    result = run_baseline_model(
        df,
        target_column="missing",
        task_type="regression",
    )

    assert result["available"] is False
