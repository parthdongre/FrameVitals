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


def test_cli_target_argument():
    parser = build_parser()

    args = parser.parse_args([
        "analyze",
        "customers.csv",
        "--target",
        "churn",
        "--mode",
        "deep",
    ])

    assert args.command == "analyze"
    assert args.target == "churn"
    assert args.mode == "deep"
