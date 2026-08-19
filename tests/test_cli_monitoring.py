import json

from framevitals.cli import build_parser, main
from framevitals.snapshots import load_snapshot


def _dataset(path):
    path.write_text(
        "value,other,group\n"
        "1,2,a\n"
        "2,4,b\n"
        "3,6,a\n"
        "4,8,b\n"
        "5,10,a\n"
        "6,12,b\n",
        encoding="utf-8",
    )


def test_snapshot_and_compare_snapshot_parsers():
    parser = build_parser()

    snapshot_args = parser.parse_args([
        "snapshot",
        "dataset.csv",
        "--mode",
        "quick",
        "--workers",
        "1",
        "--output",
        "snapshot.json",
    ])
    assert snapshot_args.command == "snapshot"
    assert snapshot_args.file.name == "dataset.csv"
    assert snapshot_args.mode == "quick"
    assert snapshot_args.workers == 1
    assert snapshot_args.output.name == "snapshot.json"

    compare_args = parser.parse_args([
        "compare-snapshots",
        "baseline.json",
        "current.json",
        "--format",
        "json",
        "--fail-on-change",
        "--output",
        "diff.json",
    ])
    assert compare_args.command == "compare-snapshots"
    assert compare_args.reference.name == "baseline.json"
    assert compare_args.current.name == "current.json"
    assert compare_args.format == "json"
    assert compare_args.fail_on_change is True
    assert compare_args.output.name == "diff.json"


def test_snapshot_cli_writes_loadable_compact_state(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset.csv"
    snapshot_path = tmp_path / "snapshot.json"
    _dataset(dataset)

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "snapshot",
            str(dataset),
            "--mode",
            "quick",
            "--workers",
            "1",
            "--format",
            "json",
            "--output",
            str(snapshot_path),
        ],
    )

    assert main() == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    saved = load_snapshot(snapshot_path)

    assert stdout_payload["snapshot_schema_version"] == "1"
    assert stdout_payload["fingerprint"] == saved["fingerprint"]
    assert saved["source"]["filename"] == "dataset.csv"
    assert saved["state"]["analysis_mode"] == "quick"
    assert saved["state"]["config"]["artifacts"] is False
    assert "profile" not in saved


def test_compare_snapshots_cli_reports_unchanged_state(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset.csv"
    snapshot_path = tmp_path / "snapshot.json"
    diff_path = tmp_path / "diff.json"
    _dataset(dataset)

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "snapshot",
            str(dataset),
            "--mode",
            "quick",
            "--workers",
            "1",
            "--format",
            "json",
            "--output",
            str(snapshot_path),
        ],
    )
    assert main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "compare-snapshots",
            str(snapshot_path),
            str(snapshot_path),
            "--format",
            "json",
            "--fail-on-change",
            "--output",
            str(diff_path),
        ],
    )
    assert main() == 0
    diff = json.loads(capsys.readouterr().out)

    assert diff["changed"] is False
    assert diff["schema"]["added_columns"] == []
    assert diff["schema"]["removed_columns"] == []
    assert json.loads(diff_path.read_text(encoding="utf-8")) == diff


def test_compare_snapshots_cli_can_fail_ci_on_change(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset.csv"
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _dataset(dataset)

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "snapshot",
            str(dataset),
            "--mode",
            "quick",
            "--workers",
            "1",
            "--output",
            str(baseline),
        ],
    )
    assert main() == 0
    capsys.readouterr()

    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["fingerprint"] = "0" * 64
    payload["state"]["dataset"]["dtypes"]["new_column"] = "int64"
    current.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "compare-snapshots",
            str(baseline),
            str(current),
            "--format",
            "terminal",
            "--fail-on-change",
        ],
    )
    assert main() == 1
    rendered = capsys.readouterr().out

    assert "FrameVitals snapshot diff" in rendered
    assert "Changed         yes" in rendered
    assert "Added columns   1" in rendered
