import pandas as pd

from framevitals.pipeline import (
    run_full_analysis,
)


def test_quick_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame({
        "age": [
            20, 21, 22, 23, 24,
        ],
        "income": [
            30000,
            32000,
            35000,
            37000,
            40000,
        ],
        "city": [
            "Pune",
            "Mumbai",
            "Pune",
            "Nashik",
            "Mumbai",
        ],
    })

    dataset_path = tmp_path / "sample.csv"

    df.to_csv(
        dataset_path,
        index=False,
    )

    result = run_full_analysis(
        dataset_id="pipeline_test",
        file_path=dataset_path,
        original_filename="sample.csv",
        analysis_mode="quick",
        skip_ai=True,
    )

    assert result["dataset_id"] == (
        "pipeline_test"
    )

    assert result["filename"] == (
        "sample.csv"
    )

    assert result["analysis_mode"] == (
        "quick"
    )

    assert result["profile"]["shape"]["rows"] == 5

    assert result["health"]["overall_score"] >= 0

    assert result["ml_readiness"]["score"] >= 0

    assert result["deep_statistics_v2"] is None
    assert result["anomalies_v2"] is None
    assert result["model_leaderboard"] is None
    assert result["explainability"] is None
    assert result["time_series"] is None
    assert result["text_profile"] is None

    assert result["charts"] == []

    assert result["ai_report"]["source"] == (
        "skipped"
    )

    assert "total" in result["timings_ms"]
    from framevitals import analyze


def test_public_api_uses_pipeline(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    path = tmp_path / "customers.csv"

    pd.DataFrame({
        "age": [20, 30, 40],
        "income": [30000, 40000, 50000],
    }).to_csv(
        path,
        index=False,
    )

    result = analyze(
        path,
        mode="quick",
    )

    assert result["filename"] == (
        "customers.csv"
    )

    assert result["analysis_mode"] == (
        "quick"
    )

    assert result["dataset_id"].startswith(
        "fv_"
    )
