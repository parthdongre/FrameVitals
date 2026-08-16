import pandas as pd

import framevitals
from framevitals.column_roles import infer_column_roles
from framevitals.target_intelligence import run_target_intelligence


def test_classification_target_intelligence_ranks_mixed_feature_types():
    target = [0, 1] * 20
    df = pd.DataFrame({
        "numeric_signal": [value * 10 + (index % 3) for index, value in enumerate(target)],
        "segment": ["stay" if value == 0 else "leave" for value in target],
        "noise": list(range(40)),
        "churn": target,
    })

    result = run_target_intelligence(
        df,
        target_column="churn",
        column_roles=infer_column_roles(df),
    )

    assert result["available"] is True
    assert result["task_type"] == "classification"
    assert result["split_guidance"]["strategy"] == "stratified_random_split"

    associations = {item["feature"]: item for item in result["top_associations"]}
    assert associations["numeric_signal"]["method"] == "point_biserial"
    assert associations["numeric_signal"]["score"] > 0.9
    assert associations["segment"]["method"] == "cramers_v"
    assert associations["segment"]["score"] > 0.9

    leakage_features = {item["feature"] for item in result["leakage"]["warnings"]}
    assert "segment" in leakage_features


def test_regression_target_intelligence_uses_spearman_and_correlation_ratio():
    target = list(range(1, 41))
    df = pd.DataFrame({
        "linear": [value * 3 for value in target],
        "bucket": ["low"] * 20 + ["high"] * 20,
        "target": target,
    })

    result = run_target_intelligence(df, target_column="target")
    associations = {item["feature"]: item for item in result["top_associations"]}

    assert result["task_type"] == "regression"
    assert associations["linear"]["method"] == "spearman"
    assert associations["linear"]["score"] == 1.0
    assert associations["bucket"]["method"] == "correlation_ratio"
    assert associations["bucket"]["score"] > 0.8


def test_time_like_columns_trigger_split_review_guidance():
    df = pd.DataFrame({
        "event_date": pd.date_range("2026-01-01", periods=30, freq="D"),
        "value": list(range(30)),
        "target": [0, 1] * 15,
    })

    result = run_target_intelligence(df, target_column="target")

    assert result["split_guidance"]["strategy"] == "review_time_aware_split"
    assert "event_date" in result["split_guidance"]["time_candidates"]


def test_target_intelligence_warns_for_identifier_like_target():
    df = pd.DataFrame({
        "feature": list(range(20)),
        "customer_id": [f"CUST-{index:04d}" for index in range(20)],
    })

    result = run_target_intelligence(df, target_column="customer_id")
    warning_codes = {item["code"] for item in result["warnings"]}

    assert "target.id_like" in warning_codes


def test_public_analyze_includes_target_intelligence_even_in_quick_mode():
    df = pd.DataFrame({
        "age": list(range(20, 40)),
        "plan": ["basic", "pro"] * 10,
        "churn": [0, 1] * 10,
    })

    result = framevitals.analyze(df, target="churn", mode="quick")

    intelligence = result["target_intelligence"]
    assert intelligence["available"] is True
    assert intelligence["target_column"] == "churn"
    assert intelligence["task_type"] == "classification"
    assert isinstance(intelligence["top_associations"], list)
    assert "target_intelligence" in result["timings_ms"]
