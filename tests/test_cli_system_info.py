import json

from framevitals.cli import build_parser, main


def test_system_info_parser_exposes_gpu_and_output_controls():
    parser = build_parser()

    args = parser.parse_args([
        "system-info",
        "--no-probe-gpu",
        "--format",
        "json",
        "--output",
        "system.json",
    ])

    assert args.command == "system-info"
    assert args.probe_gpu is False
    assert args.format == "json"
    assert args.output.name == "system.json"


def test_system_info_cli_routes_to_canonical_capability_api(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "system.json"
    expected = {
        "python": "3.12.0",
        "backend": "numpy",
        "native": {
            "available": False,
            "reason": "test fixture",
        },
        "gpu": {
            "probed": False,
            "available": False,
        },
    }
    calls = []

    def fake_system_info(*, probe_gpu=True):
        calls.append(probe_gpu)
        return expected

    monkeypatch.setattr("framevitals.acceleration.system_info", fake_system_info)
    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "system-info",
            "--no-probe-gpu",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    assert calls == [False]
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == expected
    assert saved == expected


def test_system_info_terminal_render_is_human_readable(monkeypatch, capsys):
    monkeypatch.setattr(
        "framevitals.acceleration.system_info",
        lambda *, probe_gpu=True: {
            "backend": "numpy",
            "native": {"available": True},
        },
    )
    monkeypatch.setattr("sys.argv", ["framevitals", "system-info"])

    assert main() == 0
    rendered = capsys.readouterr().out
    assert "FrameVitals system info" in rendered
    assert "Backend" in rendered
    assert "Native" in rendered
    assert "available: True" in rendered
