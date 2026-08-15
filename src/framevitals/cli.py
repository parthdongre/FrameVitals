import argparse
import json
from pathlib import Path

from framevitals import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framevitals",
        description=(
            "Automated diagnostics and ML-readiness "
            "analysis for tabular datasets."
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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command != "analyze":
        parser.print_help()
        return 0

    from framevitals.api import analyze

    report = analyze(
        args.file,
        target=args.target,
        mode=args.mode,
    )

    summary = {
        "file": report.get("filename"),
        "mode": report.get("analysis_mode"),
        "health": report.get("health"),
        "ml_readiness": report.get("ml_readiness"),
        "dataset_signals": report.get("dataset_signals"),
        "timings_ms": report.get("timings_ms"),
    }

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
