import pandas as pd

from framevitals.drift_analysis import (
    compare_datasets,
    split_by_date,
)
from framevitals.multicollinearity import (
    run_multicollinearity_analysis,
)
from framevitals.target_leakage import (
    run_target_leakage_analysis,
)


def test_numeric_drift_detection():
    reference = pd.DataFrame({
        "value": list(range(50)),
    })

    current = pd.DataFrame({
        "value": list(range(100, 150)),
    })

    result = compare_datasets(
        reference,
        current,
    )

    assert result["available"] is True

    assert result["summary"][
        "n_columns_compared"
    ] == 1

    column = result["columns"][0]

    assert column["column"] == "value"
    assert column["type"] == "numeric"
    assert column["available"] is True

    assert column["psi_severity"] in {
        "moderate",
        "severe",
    }


def test_categorical_drift_detection():
    reference = pd.DataFrame({
        "city": (
            ["Pune"] * 30
            + ["Mumbai"] * 20
        ),
    })

    current = pd.DataFrame({
        "city": (
            ["Pune"] * 5
            + ["Mumbai"] * 5
            + ["Delhi"] * 40
        ),
    })

    result = compare_datasets(
        reference,
        current,
    )

    column = result["columns"][0]

    assert column["type"] == "categorical"
    assert column["available"] is True

    assert "Delhi" in column[
        "new_categories"
    ]


def test_categorical_drift_preserves_new_and_missing_category_order():
    reference = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
    })
    current = pd.DataFrame({
        "group": ["B"] * 10 + ["C"] * 10 + ["D"] * 10 + ["E"] * 10,
    })

    result = compare_datasets(reference, current)
    column = result["columns"][0]

    assert column["new_categories"] == ["D", "E"]
    assert column["missing_categories"] == ["A"]
    assert column["n_categories_ref"] == 3
    assert column["n_categories_cur"] == 4


def test_split_by_date():
    df = pd.DataFrame({
        "date": pd.date_range(
            "2026-01-01",
            periods=40,
            freq="D",
        ),
        "value": range(40),
    })

    older, newer = split_by_date(
        df,
        "date",
    )

    assert len(older) == 20
    assert len(newer) == 20

    assert older["date"].max() < (
        newer["date"].min()
    )


def test_target_leakage_detection():
    df = pd.DataFrame({
        "feature": [
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
        ],
        "target": [
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
        ],
    })

    result = run_target_leakage_analysis(
        df,
        target_column="target",
    )

    assert result["available"] is True
    assert result["warning_count"] == 1

    warning = result["warnings"][0]

    assert warning["feature"] == "feature"
    assert warning["risk"] == "Critical"
    assert warning["same_ratio"] == 1.0


def test_multicollinearity_detection():
    x = list(range(1, 31))

    df = pd.DataFrame({
        "x1": x,
        "x2": [
            value * 2
            for value in x
        ],
        "x3": [
            value % 7
            for value in x
        ],
        "target": [
            value % 2
            for value in x
        ],
    })

    result = run_multicollinearity_analysis(
        df,
        target_column="target",
    )

    vif = result["vif"]

    assert vif["available"] is True
    assert vif["high_vif_count"] >= 2

    redundant = result[
        "redundant_groups"
    ]

    assert redundant["available"] is True
    assert redundant["group_count"] >= 1
