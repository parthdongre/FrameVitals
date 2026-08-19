from pathlib import Path

import pandas as pd
import pytest

import framevitals as fv
from framevitals.health_score import calculate_health_score
from framevitals.profiler import build_profile


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "age": list(range(20, 20 + rows)),
        "income": [30_000 + index * 1_000 for index in range(rows)],
        "city": ["Pune", "Mumbai"] * (rows // 2),
        "target": [0, 1] * (rows // 2),
    })


def test_profile_is_focused_and_matches_profiler():
    df = _frame()

    result = fv.profile(df)
    expected = build_profile(df)

    assert result["dataset_name"] == "<dataframe>"
    assert result["shape"] == expected["shape"]
    assert result["dtypes"] == expected["dtypes"]
    assert result["missing_counts"] == expected["missing_counts"]


def test_roles_health_and_ml_readiness_are_public():
    df = _frame()

    role_result = fv.roles(df)
    health_result = fv.health(df)
    readiness_result = fv.ml_readiness(df)

    assert role_result["dataset_name"] == "<dataframe>"
    assert set(role_result["columns"]) == set(df.columns)
    assert "summary" in role_result

    expected_health = calculate_health_score(df, build_profile(df))
    assert health_result["overall_score"] == expected_health["overall_score"]
    assert health_result["label"] == expected_health["label"]

    assert readiness_result["dataset_name"] == "<dataframe>"
    assert 0 <= readiness_result["score"] <= 100


def test_quality_statistics_and_anomalies_can_run_independently():
    df = _frame()

    quality = fv.quality(df, max_sample_rows=20)
    statistics = fv.statistics(df, max_pairs=5)
    anomalies = fv.anomalies(df, top_k=5)

    assert quality["dataset_name"] == "<dataframe>"
    assert quality["available"] is True
    assert quality["max_sample_rows"] == 20

    assert statistics["dataset_name"] == "<dataframe>"
    assert isinstance(statistics, dict)

    assert anomalies["dataset_name"] == "<dataframe>"
    assert anomalies["available"] is True
    assert len(anomalies["top_rows"]) <= 5


def test_target_analysis_is_focused_and_validates_target():
    df = _frame()

    result = fv.target_analysis(df, target="target")

    assert result["dataset_name"] == "<dataframe>"
    assert result["available"] is True
    assert result["target_column"] == "target"
    assert result["task_type"] == "classification"

    with pytest.raises(ValueError, match="Target column not found"):
        fv.target_analysis(df, target="does_not_exist")


def test_focused_apis_preserve_file_source_name(tmp_path):
    path = tmp_path / "customers.csv"
    _frame().to_csv(path, index=False)

    assert fv.profile(path)["dataset_name"] == "customers.csv"
    assert fv.quality(path)["dataset_name"] == "customers.csv"
    assert fv.anomalies(path)["dataset_name"] == "customers.csv"


def test_focused_apis_are_side_effect_free(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _frame()

    fv.profile(df)
    fv.roles(df)
    fv.health(df)
    fv.ml_readiness(df)
    fv.quality(df)
    fv.statistics(df, max_pairs=5)
    fv.anomalies(df, top_k=5)
    fv.target_analysis(df, target="target")

    assert not Path("cleaned").exists()
    assert not Path("static/charts").exists()
    assert not Path("reports").exists()


def test_focused_functions_are_exposed_from_top_level():
    for name in (
        "profile",
        "roles",
        "health",
        "ml_readiness",
        "quality",
        "statistics",
        "anomalies",
        "target_analysis",
    ):
        assert callable(getattr(fv, name))
        assert name in fv.__all__
