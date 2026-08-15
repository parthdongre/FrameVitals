import pandas as pd
from sklearn.linear_model import LogisticRegression

import framevitals.model_leaderboard as leaderboard_module

from framevitals.anomaly_ensemble import (
    detect_anomalies_ensemble,
)
from framevitals.explainability import (
    explain_winner,
)
from framevitals.ml_preprocessing import (
    prepare_ml_matrix,
)


def make_ml_dataset():
    rows = 40

    return pd.DataFrame({
        "customer_id": [
            f"CUST-{i}"
            for i in range(rows)
        ],
        "age": [
            20 + (i % 20)
            for i in range(rows)
        ],
        "income": [
            30000 + i * 1000
            for i in range(rows)
        ],
        "city": [
            "Pune" if i % 2 == 0 else "Mumbai"
            for i in range(rows)
        ],
        "target": [
            0 if i % 2 == 0 else 1
            for i in range(rows)
        ],
    })


def test_ml_preprocessing():
    df = make_ml_dataset()

    result = prepare_ml_matrix(
        df,
        target="target",
    )

    assert result["usable"] is True

    assert "age" in result["numeric_features"]
    assert "income" in result["numeric_features"]
    assert "city" in result["categorical_features"]

    dropped_names = {
        item["column"]
        for item in result["dropped_columns"]
    }

    assert "customer_id" in dropped_names


def test_anomaly_ensemble():
    df = pd.DataFrame({
        "x": [
            *range(29),
            1000,
        ],
        "y": [
            *range(29),
            -1000,
        ],
    })

    result = detect_anomalies_ensemble(df)

    assert result["available"] is True
    assert result["n_rows_scored"] == 30
    assert len(result["detectors_run"]) >= 1
    assert "ensemble_summary" in result


def test_model_leaderboard(monkeypatch):
    df = make_ml_dataset()

    def small_registry(class_count):
        return {
            "LogisticRegression": LogisticRegression(
                max_iter=500,
                random_state=42,
            ),
        }

    monkeypatch.setattr(
        leaderboard_module,
        "_classification_registry",
        small_registry,
    )

    result = leaderboard_module.run_model_leaderboard(
        df,
        target_column="target",
        task_type="classification",
        n_splits=2,
    )

    assert result["available"] is True
    assert result["task_type"] == "classification"
    assert result["winner"] is not None
    assert result["winner"]["model"] == "LogisticRegression"


def test_explainability_without_winner():
    df = make_ml_dataset()

    result = explain_winner(
        df,
        target_column="target",
        leaderboard_result={
            "available": False,
            "winner": None,
        },
    )

    assert result["available"] is False
