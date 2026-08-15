import argparse
import json
from pathlib import Path

from framevitals import __version__


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON result.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framevitals",
        description=(
            "Data-health diagnostics, ML-readiness analysis, drift comparison, "
            "and contract validation for tabular datasets."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"FrameVitals {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a tabular dataset.",
    )
    analyze_parser.add_argument("file", type=Path, help="Path to the dataset.")
    analyze_parser.add_argument("--target", default=None, help="Optional target column.")
    analyze_parser.add_argument(
        "--mode",
        choices=["quick", "standard", "deep", "research"],
        default="standard",
        help="Analysis depth.",
    )
    analyze_parser.add_argument(
        "--artifacts",
        action="store_true",
        help="Persist cleaned CSV/chart artifacts.",
    )
    _add_output_argument(analyze_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare reference and current datasets for drift.",
    )
    compare_parser.add_argument(
        "reference", type=Path, help="Reference/baseline dataset path."
    )
    compare_parser.add_argument("current", type=Path, help="Current dataset path.")
    compare_parser.add_argument(
        "--columns",
        default=None,
        help="Optional comma-separated columns to compare.",
    )
    compare_parser.add_argument(
        "--max-columns",
        type=int,
        default=30,
        help="Maximum number of shared columns to compare.",
    )
    _add_output_argument(compare_parser)

    contract_parser = subparsers.add_parser(
        "contract",
        help="Create or inspect reusable data-health contracts.",
    )
    contract_subparsers = contract_parser.add_subparsers(dest="contract_command")
    infer_parser = contract_subparsers.add_parser(
        "infer",
        help="Infer a contract from a known-good reference dataset.",
    )
    infer_parser.add_argument("reference", type=Path, help="Reference dataset path.")
    infer_parser.add_argument(
        "--missing-tolerance",
        type=float,
        default=0.05,
        help="Absolute missing-fraction tolerance added to the observed baseline.",
    )
    infer_parser.add_argument(
        "--duplicate-tolerance",
        type=float,
        default=0.02,
        help="Absolute duplicate-fraction tolerance added to the observed baseline.",
    )
    infer_parser.add_argument(
        "--max-allowed-values",
        type=int,
        default=20,
        help="Capture baseline category values for columns at or below this cardinality.",
    )
    infer_parser.add_argument(
        "--output",
        type=Path,
        default=Path("framevitals-contract.json"),
        help="Contract JSON path (default: framevitals-contract.json).",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a dataset against a FrameVitals contract.",
    )
    validate_parser.add_argument("file", type=Path, help="Dataset to validate.")
    validate_parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Path to a FrameVitals JSON contract.",
    )
    validate_parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return a non-zero exit code when validation has warnings.",
    )
    _add_output_argument(validate_parser)

    return parser


def _render(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _emit(payload: dict, output: Path | None) -> None:
    rendered = _render(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        from framevitals.api import analyze

        report = analyze(
            args.file,
            target=args.target,
            mode=args.mode,
            artifacts=args.artifacts,
        )
        summary = {
            "file": report.get("filename"),
            "mode": report.get("analysis_mode"),
            "health": report.get("health"),
            "ml_readiness": report.get("ml_readiness"),
            "dataset_signals": report.get("dataset_signals"),
            "artifacts_enabled": report.get("artifacts_enabled"),
            "timings_ms": report.get("timings_ms"),
        }
        _emit(summary, args.output)
        return 0

    if args.command == "compare":
        from framevitals.api import compare

        columns = None
        if args.columns:
            columns = [
                value.strip()
                for value in args.columns.split(",")
                if value.strip()
            ]

        report = compare(
            args.reference,
            args.current,
            columns=columns,
            max_columns=args.max_columns,
        )
        _emit(report, args.output)
        return 0

    if args.command == "contract" and args.contract_command == "infer":
        from framevitals.api import infer_contract
        from framevitals.contracts import save_contract

        contract = infer_contract(
            args.reference,
            missing_tolerance=args.missing_tolerance,
            duplicate_tolerance=args.duplicate_tolerance,
            max_allowed_values=args.max_allowed_values,
        )
        save_contract(contract, args.output)
        print(f"Wrote FrameVitals contract to {args.output}")
        return 0

    if args.command == "validate":
        from framevitals.api import validate

        result = validate(args.file, args.contract)
        _emit(result, args.output)
        if result["status"] == "fail":
            return 2
        if result["status"] == "warn" and args.fail_on_warn:
            return 1
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
