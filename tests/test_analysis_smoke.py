import framevitals


def test_quick_analysis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    dataset = tmp_path / "sample.csv"

    dataset.write_text(
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

    report = framevitals.analyze(
        dataset,
        mode="quick",
    )

    assert report["filename"] == "sample.csv"
    assert report["analysis_mode"] == "quick"

    assert "profile" in report
    assert "health" in report
    assert "ml_readiness" in report

    assert report["profile"]["shape"]["rows"] == 10
    assert report["profile"]["shape"]["columns"] == 3

    health_score = report["health"]["overall_score"]

    assert 0 <= health_score <= 100
