import pandas as pd

from framevitals.column_roles import infer_column_roles
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.ml_preprocessing import prepare_ml_matrix
from framevitals.ml_readiness import calculate_ml_readiness
from framevitals.profiler import build_profile, detect_column_types


def test_string_dtype_is_treated_as_categorical():
    df = pd.DataFrame(
        {
            "segment": pd.Series(["alpha", "beta"] * 10, dtype="string"),
            "value": range(20),
            "target": [0, 1] * 10,
        }
    )

    numeric, categorical, _ = detect_column_types(df)
    assert "value" in numeric
    assert "segment" in categorical

    roles = infer_column_roles(df)
    assert roles["segment"]["is_categorical"] is True
    assert "categorical" in roles["segment"]["roles"]

    readiness = calculate_ml_readiness(df)
    assert "segment" in readiness["categorical_columns"]

    prepared = prepare_ml_matrix(df, "target")
    assert "segment" in prepared["categorical_features"]


def test_string_dtype_flows_into_dataset_signals():
    df = pd.DataFrame(
        {
            "email": pd.Series(
                [f"user{i}@example.com" for i in range(20)],
                dtype="string",
            ),
            "value": range(20),
        }
    )

    profile = build_profile(df)
    signals = detect_dataset_signals(df, profile)

    assert "email" in profile["categorical_columns"]
    assert signals["has_text_columns"] is True
    assert signals["has_email_like_columns"] is True


def test_mixed_date_formats_are_detected_without_implicit_inference():
    values = ["2026-01-01", "02/02/2026", "2026-03-03", "04/04/2026"] * 5
    df = pd.DataFrame({"observed_at": pd.Series(values, dtype="string")})

    _, _, date_columns = detect_column_types(df)
    roles = infer_column_roles(df)

    assert "observed_at" in date_columns
    assert "time_like" in roles["observed_at"]["roles"]
