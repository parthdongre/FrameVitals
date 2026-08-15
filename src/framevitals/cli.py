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
        help="Optional target column.",
    )
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
