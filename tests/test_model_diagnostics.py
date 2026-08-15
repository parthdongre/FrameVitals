import pandas as pd

from framevitals.model_diagnostics import (
    run_model_diagnostics,
)


def test_classification_diagnostics():
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
        "city": [
            "Pune"
            if i % 2 == 0
            else "Mumbai"
            for i in range(rows)
        ],
        "target": [
            i % 2
            for i in range(rows)
        ],
    })

    result = run_model_diagnostics(
        df,
        target_column="target",
        task_type="classification",
    )

    assert result["available"] is True

    assert result["task_type"] == (
        "classification"
    )

    assert result["accuracy"] >= 0.0
    assert result["accuracy"] <= 1.0

    assert result["f1_weighted"] >= 0.0
    assert result["f1_weighted"] <= 1.0

    assert len(
        result["labels"]
    ) == 2

    assert len(
        result["confusion_matrix"]
    ) == 2

    assert result[
        "classification_report"
    ]

    assert result[
        "cross_validation"
    ]["f1_weighted_scores"]

    assert "customer_id" in result[
        "dropped_columns"
    ]


def test_regression_diagnostics():
    rows = 80

    df = pd.DataFrame({
        "x": list(range(rows)),
        "noise": [
            (i * 7) % 11
            for i in range(rows)
        ],
        "target": [
            (i * 3) + (i % 5)
            for i in range(rows)
        ],
    })

    result = run_model_diagnostics(
        df,
        target_column="target",
        task_type="regression",
    )

    assert result["available"] is True

    assert result["task_type"] == (
        "regression"
    )

    residuals = result[
        "residual_summary"
    ]

    assert "mean_residual" in residuals
    assert "std_residual" in residuals
    assert "mean_abs_residual" in residuals

    assert len(
        result["worst_predictions"]
    ) <= 10

    assert len(
        result["cross_validation"][
            "r2_scores"
        ]
    ) == 5


def test_model_diagnostics_invalid_target():
    df = pd.DataFrame({
        "x": list(range(40)),
    })

    result = run_model_diagnostics(
        df,
        target_column="missing",
        task_type="regression",
    )

    assert result["available"] is False


def test_model_diagnostics_unknown_task():
    df = pd.DataFrame({
        "x": list(range(40)),
        "target": list(range(40)),
    })

    result = run_model_diagnostics(
        df,
        target_column="target",
        task_type="something_else",
    )

    assert result["available"] is False
