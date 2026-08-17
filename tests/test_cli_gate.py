import json

from framevitals.cli import build_parser, main


def test_gate_parser_exposes_ci_controls():
    parser = build_parser()
    args = parser.parse_args([
        "gate",
        "production.csv",
        "--reference",
        "training.csv",
        "--contract",
        "contract.json",
        "--columns",
        "age,income",
        "--max-columns",
        "12",
        "--drift-warn-on",
        "minor",
        "--drift-fail-on",
        "moderate",
        "--fail-on-validation-warning",
        "--format",
        "json",
        "--output",
        "gate.json",
    ])

    assert args.command == "gate"
    assert args.current.name == "production.csv"
    assert args.reference.name == "training.csv"
    assert args.contract.name == "contract.json"
    assert args.columns == "age,income"
    assert args.max_columns == 12
    assert args.drift_warn_on == "minor"
    assert args.drift_fail_on == "moderate"
    assert args.fail_on_validation_warning is True
    assert args.format == "json"
    assert args.output.name == "gate.json"


def test_gate_cli_returns_zero_for_passing_contract(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "candidate.csv"
    contract = tmp_path / "contract.json"
    dataset.write_text("age,plan\n25,basic\n30,pro\n", encoding="utf-8")
    contract.write_text(
        json.dumps({
            "version": 2,
            "allow_extra_columns": False,
            "columns": {
                "age": {
                    "type": "integer",
                    "nullable": False,
                    "minimum": 18,
                    "maximum": 100,
                },
                "plan": {
                    "type": "string",
                    "nullable": False,
                    "allowed_values": ["basic", "pro"],
                },
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "gate",
            str(dataset),
            "--contract",
            str(contract),
            "--format",
            "json",
        ],
    )

    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert payload["passed"] is True
    assert payload["checks_run"] == ["validation"]


def test_gate_cli_returns_one_for_failing_contract(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "candidate.csv"
    contract = tmp_path / "contract.json"
    dataset.write_text("age\n15\n", encoding="utf-8")
    contract.write_text(
        json.dumps({
            "version": 2,
            "allow_extra_columns": False,
            "columns": {
                "age": {
                    "type": "integer",
                    "nullable": False,
                    "minimum": 18,
                    "maximum": 100,
                },
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "gate",
            str(dataset),
            "--contract",
            str(contract),
            "--format",
            "json",
        ],
    )

    assert main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["passed"] is False
    assert payload["checks"]["validation"]["status"] == "fail"
