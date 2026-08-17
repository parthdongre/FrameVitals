import json

from framevitals.cli import build_parser, main


def test_cli_parser():
    parser = build_parser()

    args = parser.parse_args([
        "analyze",
        "dataset.csv",
        "--mode",
        "quick",
    ])

    assert args.command == "analyze"
    assert args.file.name == "dataset.csv"
    assert args.mode == "quick"
    assert args.target is None
    assert args.artifacts is None
    assert args.workers is None
    assert args.preset is None
    assert args.config is None
    assert args.output is None


def test_cli_inspect_parser():
    parser = build_parser()

    args = parser.parse_args([
        "inspect",
        "dataset.parquet",
        "--format",
        "json",
        "--output",
        "source.json",
    ])

    assert args.command == "inspect"
    assert args.file.name == "dataset.parquet"
    assert args.format == "json"
    assert args.output.name == "source.json"


def test_cli_inspect_emits_source_metadata_and_json_file(
    tmp_path,
    monkeypatch,
    capsys,
):
    dataset = tmp_path / "dataset.csv"
    output = tmp_path / "source.json"
    dataset.write_text("value,label\n1,a\n2,b\n3,c\n", encoding="utf-8")

    monkeypatch.setattr(
        "framevitals.sources.DelimitedTextSource._pyarrow_csv",
        lambda self: None,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "inspect",
            str(dataset),
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    rendered = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert rendered == saved
    assert rendered["name"] == "dataset.csv"
    assert rendered["kind"] == "file"
    assert rendered["format"] == "csv"
    assert rendered["supports_streaming"] is False
    assert rendered["supports_projection"] is False
    assert rendered["size_bytes"] == dataset.stat().st_size


def test_cli_target_argument():
    parser = build_parser()

    args = parser.parse_args([
        "analyze",
        "customers.csv",
        "--target",
        "churn",
        "--mode",
        "deep",
        "--artifacts",
        "--workers",
        "6",
        "--output",
        "result.json",
    ])

    assert args.command == "analyze"
    assert args.target == "churn"
    assert args.mode == "deep"
    assert args.artifacts is True
    assert args.workers == 6
    assert args.output.name == "result.json"


def test_cli_config_and_preset_arguments():
    parser = build_parser()

    args = parser.parse_args([
        "analyze",
        "customers.csv",
        "--preset",
        "ci",
        "--config",
        "framevitals.toml",
        "--no-artifacts",
    ])
    config_args = parser.parse_args([
        "config",
        "--file",
        "framevitals.toml",
        "--preset",
        "deep",
    ])

    assert args.preset == "ci"
    assert args.config.name == "framevitals.toml"
    assert args.artifacts is False
    assert config_args.command == "config"
    assert config_args.file.name == "framevitals.toml"
    assert config_args.preset == "deep"


def test_cli_compare_parser_supports_gate_controls():
    parser = build_parser()

    args = parser.parse_args([
        "compare",
        "train.csv",
        "production.csv",
        "--columns",
        "age,income",
        "--max-columns",
        "12",
        "--format",
        "terminal",
        "--fail-on",
        "moderate",
        "--output",
        "drift.json",
    ])

    assert args.command == "compare"
    assert args.reference.name == "train.csv"
    assert args.current.name == "production.csv"
    assert args.columns == "age,income"
    assert args.max_columns == 12
    assert args.format == "terminal"
    assert args.fail_on == "moderate"
    assert args.output.name == "drift.json"


def test_cli_contract_parsers_expose_inference_and_warning_controls():
    parser = build_parser()

    infer_args = parser.parse_args([
        "infer-contract",
        "reference.csv",
        "--numeric-tolerance",
        "0.1",
        "--max-categories",
        "15",
        "--null-fraction-tolerance",
        "0.08",
        "--no-infer-unique",
        "--allow-extra-columns",
        "--output",
        "contract.json",
    ])
    validate_args = parser.parse_args([
        "validate",
        "candidate.csv",
        "--contract",
        "contract.json",
        "--format",
        "terminal",
        "--fail-on-warn",
        "--output",
        "validation.json",
    ])

    assert infer_args.command == "infer-contract"
    assert infer_args.file.name == "reference.csv"
    assert infer_args.numeric_tolerance == 0.1
    assert infer_args.max_categories == 15
    assert infer_args.null_fraction_tolerance == 0.08
    assert infer_args.infer_unique is False
    assert infer_args.allow_extra_columns is True
    assert infer_args.output.name == "contract.json"
    assert validate_args.command == "validate"
    assert validate_args.file.name == "candidate.csv"
    assert validate_args.contract.name == "contract.json"
    assert validate_args.format == "terminal"
    assert validate_args.fail_on_warn is True
    assert validate_args.output.name == "validation.json"


def test_cli_validate_returns_two_for_contract_errors(tmp_path, monkeypatch):
    dataset = tmp_path / "candidate.csv"
    contract = tmp_path / "contract.json"
    dataset.write_text("age\n15\n", encoding="utf-8")
    contract.write_text(
        json.dumps({
            "version": 1,
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
            "validate",
            str(dataset),
            "--contract",
            str(contract),
        ],
    )

    assert main() == 2


def test_cli_validate_warning_exit_is_opt_in(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "candidate.csv"
    contract = tmp_path / "contract.json"
    dataset.write_text("plan\nenterprise\n", encoding="utf-8")
    contract.write_text(
        json.dumps({
            "version": 2,
            "columns": {
                "plan": {
                    "type": "string",
                    "nullable": False,
                    "allowed_values": ["basic", "pro"],
                    "allowed_values_severity": "warning",
                },
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["framevitals", "validate", str(dataset), "--contract", str(contract)],
    )
    assert main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "validate",
            str(dataset),
            "--contract",
            str(contract),
            "--fail-on-warn",
        ],
    )
    assert main() == 1
