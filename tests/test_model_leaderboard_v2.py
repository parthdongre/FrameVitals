import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge

import framevitals.model_leaderboard as leaderboard_module


def _classification_frame(labels=(10, 20), rows=40):
    return pd.DataFrame({
        "feature": list(range(rows)),
        "segment": ["A", "B"] * (rows // 2),
        "target": [labels[index % len(labels)] for index in range(rows)],
    })


def test_non_consecutive_numeric_class_labels_are_encoded_safely(monkeypatch):
    def registry(class_count):
        return {
            "DummyClassifier": DummyClassifier(strategy="most_frequent"),
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        }

    monkeypatch.setattr(leaderboard_module, "_classification_registry", registry)
    result = leaderboard_module.run_model_leaderboard(
        _classification_frame(labels=(10, 20)),
        target_column="target",
        task_type="classification",
        n_splits=2,
    )

    assert result["available"] is True
    assert result["target_encoding"] == [
        {"encoded": 0, "label": 10},
        {"encoded": 1, "label": 20},
    ]
    assert result["winner"] is not None
    assert result["models_succeeded"] >= 2
    assert result["baseline"]["model"] == "DummyClassifier"


def test_rare_class_returns_clear_cv_unavailable_result(monkeypatch):
    df = pd.DataFrame({
        "feature": list(range(25)),
        "target": [0] * 24 + [1],
    })

    def registry(class_count):
        return {"LogisticRegression": LogisticRegression(max_iter=200)}

    monkeypatch.setattr(leaderboard_module, "_classification_registry", registry)
    result = leaderboard_module.run_model_leaderboard(
        df,
        target_column="target",
        task_type="classification",
        n_splits=5,
    )

    assert result["available"] is False
    assert "at least 2 rows" in result["message"]


def test_regression_cv_caps_requested_folds_and_reports_baseline(monkeypatch):
    rows = 24
    df = pd.DataFrame({
        "feature": list(range(rows)),
        "target": [float(index * 3 + 1) for index in range(rows)],
    })

    def registry():
        return {
            "DummyRegressor": DummyRegressor(strategy="mean"),
            "Ridge": Ridge(alpha=1.0),
        }

    monkeypatch.setattr(leaderboard_module, "_regression_registry", registry)
    result = leaderboard_module.run_model_leaderboard(
        df,
        target_column="target",
        task_type="regression",
        n_splits=100,
    )

    assert result["available"] is True
    assert result["cv"]["requested_splits"] == 100
    # R2 needs at least two observations in each test fold, so the safe upper
    # bound is floor(n_rows / 2), not one fold per row.
    assert result["cv"]["actual_splits"] == rows // 2
    assert result["baseline"]["model"] == "DummyRegressor"
    assert result["winner"]["model"] == "Ridge"
    assert result["winner"]["beats_baseline"] is True
    assert result["winner"]["lift_over_baseline"] is not None


def test_leaderboard_records_model_failures_without_sinking_other_models(monkeypatch):
    class BrokenEstimator:
        def get_params(self, deep=True):
            return {}

        def fit(self, X, y):
            raise RuntimeError("broken intentionally")

    def registry(class_count):
        return {
            "Broken": BrokenEstimator(),
            "LogisticRegression": LogisticRegression(max_iter=500),
        }

    monkeypatch.setattr(leaderboard_module, "_classification_registry", registry)
    result = leaderboard_module.run_model_leaderboard(
        _classification_frame(labels=(0, 1)),
        target_column="target",
        task_type="classification",
        n_splits=2,
    )

    assert result["available"] is True
    assert result["winner"]["model"] == "LogisticRegression"
    assert result["models_failed"] == 1
    assert result["model_failures"][0]["model"] == "Broken"


def test_leaderboard_validates_controls():
    df = _classification_frame(labels=(0, 1))

    with pytest.raises(ValueError, match="task_type"):
        leaderboard_module.run_model_leaderboard(
            df,
            target_column="target",
            task_type="ranking",
        )
    with pytest.raises(ValueError, match="n_splits"):
        leaderboard_module.run_model_leaderboard(
            df,
            target_column="target",
            n_splits=1,
        )
    with pytest.raises(ValueError, match="n_jobs"):
        leaderboard_module.run_model_leaderboard(
            df,
            target_column="target",
            n_jobs=0,
        )
