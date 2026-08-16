import argparse
import json
from pathlib import Path

from framevitals import __version__
from framevitals.config import available_presets


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the complete JSON result.",
    )


def _load_contract(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Contract file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Contract file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Contract JSON must contain an object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framevitals",
        description=(
            "Automated diagnostics, ML-readiness analysis, and drift "
            "comparison for tabular datasets."
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
    analyze_parser.add_argument(
        "file",
        type=Path,
        help="Path to the dataset.",
    )
    analyze_parser.add_argument(
        "--target",
        default=None,
        help="Optional target column. Overrides config values.",
    )
    analyze_parser.add_argument(
        "--mode",
        choices=["quick", "standard", "deep", "research"],
        default=None,
        help="Analysis depth. Overrides preset/config values.",
    )
    analyze_parser.add_argument(
        "--preset",
        choices=list(available_presets()),
        default=None,
        help="Built-in runtime preset.",
    )
    analyze_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional FrameVitals TOML config path.",
    )
    analyze_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Overrides config values.",
    )
    analyze_parser.add_argument(
        "--artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable cleaned CSV and chart artifacts.",
    )
    analyze_parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Stdout format. JSON prints the complete result.",
    )
    analyze_parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help="Optional path to write a self-contained HTML report.",
    )
    _add_output_argument(analyze_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare reference and current datasets for drift.",
    )
    compare_parser.add_argument(
        "reference",
        type=Path,
        help="Reference/baseline dataset path.",
    )
    compare_parser.add_argument(
        "current",
        type=Path,
        help="Current dataset path.",
    )
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

    infer_contract_parser = subparsers.add_parser(
        "infer-contract",
        help="Infer a reusable data contract from a reference dataset.",
    )
    infer_contract_parser.add_argument(
        "file",
        type=Path,
        help="Path to the reference dataset.",
    )
    _add_output_argument(infer_contract_parser)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a dataset against a JSON data contract.",
    )
    validate_parser.add_argument(
        "file",
        type=Path,
        help="Path to the dataset to validate.",
    )
    validate_parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Path to a JSON contract created by infer-contract.",
    )
    _add_output_argument(validate_parser)

    config_parser = subparsers.add_parser(
        "config",
        help="Resolve and inspect FrameVitals runtime configuration.",
    )
    config_parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional TOML config file.",
    )
    config_parser.add_argument(
        "--preset",
        choices=list(available_presets()),
        default=None,
        help="Optional built-in preset to resolve before the config file.",
    )

    return parser


def _emit(payload: dict, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, default=str)
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
            workers=args.workers,
            preset=args.preset,
            config=args.config,
        )

        if args.output is not None:
            report.to_json(args.output)
        if args.html_report is not None:
            report.to_html(args.html_report)

        if args.format == "json":
            print(report.to_json())
        else:
            print(report.summary_text())
            resolved = report.get("config", {})
            if resolved:
                print(
                    "Config        "
                    f"mode={resolved.get('mode')} "
                    f"workers={resolved.get('workers')} "
                    f"artifacts={resolved.get('artifacts')}"
                )
            if args.output is not None:
                print(f"Full JSON     {args.output}")
            if args.html_report is not None:
                print(f"HTML report   {args.html_report}")
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

    if args.command == "infer-contract":
        from framevitals.api import infer_contract

        report = infer_contract(args.file)
        _emit(report, args.output)
        return 0

    if args.command == "validate":
        from framevitals.api import validate

        report = validate(
            args.file,
            _load_contract(args.contract),
        )
        _emit(report, args.output)
        return 0 if report["valid"] else 1

    if args.command == "config":
        from framevitals.config import resolve_config

        resolved = resolve_config(args.file, preset=args.preset)
        print(json.dumps(resolved.to_dict(), indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
