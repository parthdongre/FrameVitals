import json

from framevitals.cli import main


def _write_dataset(path):
    path.write_text(
        "age,city\n"
        "20,Pune\n"
        "30,Mumbai\n"
        "40,Pune\n"
        "50,Nashik\n",
        encoding="utf-8",
    )


def test_analyze_cli_writes_full_json_and_html(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "customers.csv"
    output = tmp_path / "report.json"
    html = tmp_path / "report.html"
    _write_dataset(dataset)

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "analyze",
            str(dataset),
            "--mode",
            "quick",
            "--output",
            str(output),
            "--html-report",
            str(html),
        ],
    )

    assert main() == 0
    stdout = capsys.readouterr().out
    assert "FrameVitals Analysis" in stdout
    assert "Full JSON" in stdout
    assert "HTML report" in stdout

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["filename"] == "customers.csv"
    assert payload["profile"]["shape"]["rows"] == 4
    assert payload["result_schema_version"] == "1"
    assert "findings" in payload

    html_text = html.read_text(encoding="utf-8")
    assert "FrameVitals analysis report" in html_text
    assert "customers.csv" in html_text


def test_analyze_cli_json_stdout_is_complete(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "customers.csv"
    _write_dataset(dataset)

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "analyze",
            str(dataset),
            "--mode",
            "quick",
            "--format",
            "json",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["columns"] == ["age", "city"]
    assert "health" in payload
    assert "ml_readiness" in payload
    assert "findings" in payload
