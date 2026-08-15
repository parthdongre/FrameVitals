import framevitals


def _write_sample_dataset(path):
    path.write_text(
        "age,income,city\n"
        "21,30000,Pune\n"
        "22,32000,Mumbai\n"
        "23,35000,Pune\n"
        "24,37000,Nashik\n"
        "25,40000,Mumbai\n"
        "26,42000,Pune\n"
        "27,45000,Nashik\n"
        "28,47000,Mumbai\n"
        "29,50000,Pune\n"
        "30,52000,Mumbai\n"
    )


def test_quick_analysis_is_side_effect_free_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "sample.csv"
    _write_sample_dataset(dataset)

    report = framevitals.analyze(dataset, mode="quick")

    assert report["filename"] == "sample.csv"
    assert report["analysis_mode"] == "quick"
    assert report["artifacts_enabled"] is False

    assert "profile" in report
    assert "health" in report
    assert "ml_readiness" in report

    assert report["profile"]["shape"]["rows"] == 10
    assert report["profile"]["shape"]["columns"] == 3

    health_score = report["health"]["overall_score"]
    assert 0 <= health_score <= 100

    cleaning = report["cleaning"]
    assert cleaning["missing_after"] == 0
    assert cleaning["output_path"] is None
    assert not (tmp_path / "cleaned").exists()


def test_quick_analysis_can_write_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "sample.csv"
    _write_sample_dataset(dataset)

    report = framevitals.analyze(
        dataset,
        mode="quick",
        artifacts=True,
    )

    assert report["artifacts_enabled"] is True
    cleaned_path = tmp_path / report["cleaning"]["output_path"]
    assert cleaned_path.exists()
