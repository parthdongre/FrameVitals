from framevitals.cli import build_parser


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
    assert args.artifacts is False
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
        "--output",
        "result.json",
    ])

    assert args.command == "analyze"
    assert args.target == "churn"
    assert args.mode == "deep"
    assert args.artifacts is True
    assert args.output.name == "result.json"


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


def test_cli_contract_infer_parser():
    parser = build_parser()

    args = parser.parse_args([
        "contract",
        "infer",
        "train.csv",
        "--missing-tolerance",
        "0.1",
        "--output",
        "contract.json",
    ])

    assert args.command == "contract"
    assert args.contract_command == "infer"
    assert args.reference.name == "train.csv"
    assert args.missing_tolerance == 0.1
    assert args.output.name == "contract.json"


def test_cli_validate_parser():
    parser = build_parser()

    args = parser.parse_args([
        "validate",
        "batch.csv",
        "--contract",
        "contract.json",
        "--fail-on-warn",
        "--output",
        "validation.json",
    ])

    assert args.command == "validate"
    assert args.file.name == "batch.csv"
    assert args.contract.name == "contract.json"
    assert args.fail_on_warn is True
    assert args.output.name == "validation.json"
