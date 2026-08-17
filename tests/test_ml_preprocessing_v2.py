import numpy as np
import pandas as pd
import pytest

from framevitals.ml_preprocessing import prepare_ml_matrix


def test_id_detection_uses_boundaries_not_raw_substrings():
    rows = 30
    df = pd.DataFrame({
        "paid_amount": [100 + index for index in range(rows)],
        "width": [10 + (index % 5) for index in range(rows)],
        "customer_id": [f"CUST-{index:03d}" for index in range(rows)],
        "candidate_score": [50 + (index % 10) for index in range(rows)],
        "target": [0, 1] * 15,
    })

    result = prepare_ml_matrix(df, target="target")
    dropped = {item["column"]: item["reason"] for item in result["dropped_columns"]}

    assert "paid_amount" in result["numeric_features"]
    assert "width" in result["numeric_features"]
    assert "candidate_score" in result["numeric_features"]
    assert dropped["customer_id"] == "id_like_name"


def test_numeric_infinities_are_replaced_for_imputation():
    rows = 30
    df = pd.DataFrame({
        "measurement": [float(index) for index in range(rows)],
        "target": [0, 1] * 15,
    })
    df.loc[3, "measurement"] = np.inf
    df.loc[7, "measurement"] = -np.inf

    result = prepare_ml_matrix(df, target="target")

    assert result["usable"] is True
    assert result["infinite_values_replaced"] == {"measurement": 2}
    assert int(result["X"]["measurement"].isna().sum()) == 2
    assert any("infinite values" in warning for warning in result["warnings"])


def test_infinite_numeric_targets_are_dropped_as_invalid_labels():
    rows = 30
    target = [float(index) for index in range(rows)]
    target[-1] = np.inf
    df = pd.DataFrame({
        "feature": list(range(rows)),
        "target": target,
    })

    result = prepare_ml_matrix(df, target="target")

    assert result["target_infinite_values_dropped"] == 1
    assert len(result["y"]) == 29
    assert np.isfinite(result["y"].to_numpy(dtype=float)).all()


def test_high_cardinality_categorical_is_bounded_before_one_hot_encoding():
    rows = 300
    df = pd.DataFrame({
        "category": [f"group-{index % 250}" for index in range(rows)],
        "feature": [index % 12 for index in range(rows)],
        "target": [0, 1] * 150,
    })

    result = prepare_ml_matrix(
        df,
        target="target",
        max_categorical_levels=100,
    )
    dropped = {item["column"]: item["reason"] for item in result["dropped_columns"]}

    assert dropped["category"] == "high_cardinality_categorical"
    assert "category" not in result["categorical_features"]
    assert "feature" in result["numeric_features"]


def test_real_time_columns_are_excluded_but_unrelated_names_are_kept():
    rows = 30
    df = pd.DataFrame({
        "event_date": pd.date_range("2026-01-01", periods=rows).astype(str),
        "candidate_label": ["A", "B", "C"] * 10,
        "target": [0, 1] * 15,
    })

    result = prepare_ml_matrix(df, target="target")
    dropped = {item["column"]: item["reason"] for item in result["dropped_columns"]}

    assert dropped["event_date"] == "time_like_non_numeric"
    assert "candidate_label" in result["categorical_features"]


def test_preprocessing_control_validation():
    df = pd.DataFrame({"x": list(range(20)), "target": [0, 1] * 10})

    with pytest.raises(ValueError, match="drop_high_unique_ratio"):
        prepare_ml_matrix(df, "target", drop_high_unique_ratio=0)
    with pytest.raises(ValueError, match="min_non_missing"):
        prepare_ml_matrix(df, "target", min_non_missing=0)
    with pytest.raises(ValueError, match="max_categorical_levels"):
        prepare_ml_matrix(df, "target", max_categorical_levels=1)
