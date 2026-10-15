"""Unit coverage for the dependency-free, Mole-inspired terminal workspace."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from framevitals.cli import build_parser, main
from framevitals.terminal_ui import (
    ACTIONS,
    TerminalReport,
    TerminalState,
    _findings,
    _path_source,
    _text_lines,
    launch_terminal,
    run_diagnostic,
    save_report,
    view_lines,
)


class FakeProtocol:
    status = "complete"
    beacons = [
        {
            "code": "quality.missing",
            "title": "Missing data",
            "severity": "medium",
            "summary": "Two cells are missing",
            "recommendation": "Inspect source records",
        }
    ]

    def summary_text(self):
        return "FrameVitals / PRISM\n2 rows, 2 columns"

    def to_json(self):
        return '{"protocol":"prism","status":"complete"}'

    def to_html(self, destination):
        Path(destination).write_text("<html>Prism</html>", encoding="utf-8")


def test_tui_aliases_and_existing_cli_parser_still_work():
    parser = build_parser()

    assert parser.parse_args(["tui"]).command == "tui"
    args = parser.parse_args(["ui", "--plain"])
    assert args.command == "ui"
    assert args.plain is True

    old = parser.parse_args(["prism", "dataset.csv", "--depth", "quick"])
    assert old.command == "prism"
    assert old.mode == "quick"
    assert len(ACTIONS) == 6


def test_terminal_launcher_returns_help_in_noninteractive_context(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["framevitals"])
    assert main() == 0
    assert "prism" in capsys.readouterr().out.lower()


def test_terminal_ui_explicit_command_reaches_launcher(monkeypatch):
    import framevitals.terminal_ui as terminal_ui

    modes = []
    monkeypatch.setattr(
        terminal_ui,
        "launch_terminal",
        lambda *, plain=False: modes.append(plain) or 0,
    )
    monkeypatch.setattr(sys, "argv", ["framevitals", "ui", "--plain"])

    assert main() == 0
    assert modes == [True]


def test_terminal_refuses_noninteractive_tty(capsys):
    assert launch_terminal(plain=True) == 2
    assert "terminal" in capsys.readouterr().err.lower()


def test_source_loader_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        _path_source(tmp_path / "not-found.csv")


def test_numpy_source_is_loaded_without_pickle(tmp_path):
    safe = tmp_path / "matrix.npy"
    matrix = np.eye(5)
    np.save(safe, matrix)

    array = _path_source(safe)
    assert isinstance(array, np.ndarray)
    np.testing.assert_array_equal(array, matrix)

    unsafe = tmp_path / "objects.npy"
    np.save(unsafe, np.array([{"a": 1}], dtype=object))
    with pytest.raises((ValueError, TypeError)):
        _path_source(unsafe)


def test_nested_json_source_uses_parsed_data_not_unrecognized_file(tmp_path, monkeypatch):
    from framevitals import protocols

    path = tmp_path / "nested.json"
    path.write_text('{"records":[{"a":1},{"a":2}]}', encoding="utf-8")
    original = path.read_bytes()
    called = {}

    def fake_prism(data, *, reference, depth, artifacts):
        called.update(data=data, reference=reference, depth=depth, artifacts=artifacts)
        return FakeProtocol()

    monkeypatch.setattr(protocols, "prism", fake_prism)
    report = run_diagnostic("Prism", path, depth="deep")

    assert called["data"]["records"][0]["a"] == 1
    assert called["reference"] is None
    assert called["depth"] == "deep"
    assert called["artifacts"] is False
    assert report.action == "Prism"
    assert report.findings[0]["code"] == "quality.missing"
    assert "2 rows" in report.summary
    assert path.read_bytes() == original


def test_tide_requires_a_reference(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reference"):
        run_diagnostic("Tide", path)


def test_tide_passes_two_sources_in_correct_order(tmp_path, monkeypatch):
    from framevitals import protocols

    baseline = tmp_path / "baseline.csv"
    current = tmp_path / "current.csv"
    baseline.write_text("x\n1\n", encoding="utf-8")
    current.write_text("x\n2\n", encoding="utf-8")
    called = {}

    def fake_tide(reference, data):
        called["reference"] = reference
        called["current"] = data
        return {"status": "warn", "findings": []}

    monkeypatch.setattr(protocols, "tide", fake_tide)
    report = run_diagnostic("Tide", current, reference=baseline)

    assert called["reference"] == baseline
    assert called["current"] == current
    assert report.status == "warn"


def test_forge_is_always_preview_only(tmp_path, monkeypatch):
    from framevitals import protocols

    data = tmp_path / "data.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    seen = []

    def fake_forge(source, *, apply):
        seen.append((source, apply))
        return FakeProtocol()

    monkeypatch.setattr(protocols, "forge", fake_forge)
    report = run_diagnostic("Forge preview", data)

    assert report.action == "Forge preview"
    assert seen == [(data, False)]


def test_explicit_export_json_and_html(tmp_path):
    report = TerminalReport(
        action="Prism",
        result=FakeProtocol(),
        summary="summary",
    )

    destination = tmp_path / "report.json"
    assert not destination.exists()
    save_report(report, destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["protocol"] == "prism"

    html = tmp_path / "report.html"
    save_report(report, html)
    assert "<html>" in html.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="must end"):
        save_report(report, tmp_path / "report.txt")


def test_view_navigation_and_wrapping_do_not_mutate_report():
    report = TerminalReport(
        action="Prism",
        result=FakeProtocol(),
        summary="OVER\nview",
        findings=FakeProtocol.beacons,
        detail='{"hello":"world"}',
    )
    state = TerminalState(report=report)
    assert view_lines(state) == ["OVER", "view"]

    state.page = 1
    assert "Missing data" in "\n".join(view_lines(state))
    assert "Inspect source records" in "\n".join(view_lines(state))

    state.page = 2
    assert '"hello"' in view_lines(state)[0]
    assert len(_text_lines(["a" * 200], 40)) > 1
    assert _findings(FakeProtocol()) == FakeProtocol.beacons
