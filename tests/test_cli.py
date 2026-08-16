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


def test_cli_compare_parser():
    parser = build_parser()

    args = parser.parse_args([
        "compare",
        "train.csv",
        "production.csv",
        "--columns",
        "age,income",
        "--max-columns",
        "12",
        "--output",
        "drift.json",
    ])

    assert args.command == "compare"
    assert args.reference.name == "train.csv"
    assert args.current.name == "production.csv"
    assert args.columns == "age,income"
    assert args.max_columns == 12
    assert args.output.name == "drift.json"


def test_cli_contract_parsers():
    parser = build_parser()

    infer_args = parser.parse_args([
        "infer-contract",
        "reference.csv",
        "--output",
        "contract.json",
    ])
    validate_args = parser.parse_args([
        "validate",
        "candidate.csv",
        "--contract",
        "contract.json",
        "--output",
        "validation.json",
    ])

    assert infer_args.command == "infer-contract"
    assert infer_args.file.name == "reference.csv"
    assert infer_args.output.name == "contract.json"
    assert validate_args.command == "validate"
    assert validate_args.file.name == "candidate.csv"
    assert validate_args.contract.name == "contract.json"
    assert validate_args.output.name == "validation.json"


def test_cli_validate_returns_nonzero_for_contract_errors(tmp_path, monkeypatch):
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

    assert main() == 1
