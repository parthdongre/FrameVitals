import argparse
import json
from pathlib import Path

from framevitals import __version__
from framevitals.config import available_modules, available_presets


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the complete JSON result.",
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target",
        default=None,
        help="Optional target column. Overrides config values.",
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "deep", "research"],
        default=None,
        help="Analysis depth. Overrides preset/config values.",
    )
    parser.add_argument(
        "--preset",
        choices=list(available_presets()),
        default=None,
        help="Built-in runtime preset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional FrameVitals TOML config path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Overrides config values.",
    )
    parser.add_argument(
        "--disable-module",
        dest="disabled_modules",
        action="append",
        choices=list(available_modules()),
        default=None,
        help=(
            "Disable an optional execution module. Repeat the flag to disable "
            "multiple modules. Explicit flags override the config/preset list."
        ),
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


def _parse_columns(value: str | None) -> list[str] | None:
    if not value:
        return None
    columns = [item.strip() for item in value.split(",") if item.strip()]
    return columns or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framevitals",
        description=(
            "Automated diagnostics, ML-readiness analysis, validation, and drift "
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
    analyze_parser.add_argument("file", type=Path, help="Path to the dataset.")
    _add_runtime_arguments(analyze_parser)
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

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect dataset source metadata and execution capabilities.",
    )
    inspect_parser.add_argument("file", type=Path, help="Path to the dataset.")
    inspect_parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Source metadata output format.",
    )
    _add_output_argument(inspect_parser)

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Analyze a dataset and emit compact monitoring state.",
    )
    snapshot_parser.add_argument("file", type=Path, help="Path to the dataset.")
    _add_runtime_arguments(snapshot_parser)
    snapshot_parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Snapshot output format.",
    )
    _add_output_argument(snapshot_parser)

    compare_snapshots_parser = subparsers.add_parser(
        "compare-snapshots",
        help="Compare two compact FrameVitals monitoring snapshots.",
    )
    compare_snapshots_parser.add_argument(
        "reference", type=Path, help="Reference/baseline snapshot JSON."
    )
    compare_snapshots_parser.add_argument(
        "current", type=Path, help="Current snapshot JSON."
    )
    compare_snapshots_parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Snapshot-diff output format.",
    )
    compare_snapshots_parser.add_argument(
        "--fail-on-change",
        action="store_true",
        help="Return exit code 1 when the compact snapshot state changed.",
    )
    _add_output_argument(compare_snapshots_parser)

    system_info_parser = subparsers.add_parser(
        "system-info",
        help="Inspect FrameVitals CPU/native/GPU execution capabilities.",
    )
    system_info_parser.add_argument(
        "--probe-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe GPU/CUDA availability. Use --no-probe-gpu for support-safe output.",
    )
    system_info_parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="System-information output format.",
    )
    _add_output_argument(system_info_parser)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Preview applicable analyses without running heavy stages.",
    )
    plan_parser.add_argument("file", type=Path, help="Path to the dataset.")
    _add_runtime_arguments(plan_parser)
    plan_parser.add_argument(
        "--format",
        choices=["terminal", "json"],
        default="terminal",
        help="Plan output format.",
    )
    _add_output_argument(plan_parser)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Inspect a conservative cleaning plan and optionally write cleaned CSV.",
    )
    clean_parser.add_argument("file", type=Path, help="Path to the dataset.")
    clean_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the cleaned dataset to this CSV path. Omit for plan-only mode.",
    )
    clean_parser.add_argument(
        "--plan-output",
        type=Path,
        default=None,
        help="Optional path to write the inferred cleaning plan as JSON.",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare reference and current datasets for schema/distribution drift.",
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
    compare_parser.add_argument(
        "--format",
        choices=["json", "terminal"],
        default="json",
        help="Stdout format. JSON remains the default for backward compatibility.",
    )
    compare_parser.add_argument(
        "--fail-on",
        choices=["minor", "moderate", "severe"],
        default=None,
        help="Return exit code 1 when drift reaches this severity. Disabled by default.",
    )
    _add_output_argument(compare_parser)

    infer_contract_parser = subparsers.add_parser(
        "infer-contract",
        help="Infer a reusable data contract from a reference dataset.",
    )
    infer_contract_parser.add_argument(
        "file", type=Path, help="Path to the reference dataset."
    )
    infer_contract_parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=0.05,
        help="Expand inferred numeric bounds by this fraction of the observed span.",
    )
    infer_contract_parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Infer allowed-value expectations up to this cardinality.",
    )
    infer_contract_parser.add_argument(
        "--null-fraction-tolerance",
        type=float,
        default=0.05,
        help="Additional tolerated null fraction above the reference rate.",
    )
    infer_contract_parser.add_argument(
        "--infer-unique",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Infer uniqueness constraints for sufficiently large fully unique columns.",
    )
    infer_contract_parser.add_argument(
        "--min-unique-rows",
        type=int,
        default=20,
        help="Minimum non-null rows required before inferring uniqueness.",
    )
    infer_contract_parser.add_argument(
        "--allow-extra-columns",
        action="store_true",
        help="Allow columns not present in the reference contract.",
    )
    _add_output_argument(infer_contract_parser)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a dataset against a JSON data contract.",
    )
    validate_parser.add_argument(
        "file", type=Path, help="Path to the dataset to validate."
    )
    validate_parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Path to a JSON contract created by infer-contract.",
    )
    validate_parser.add_argument(
        "--format",
        choices=["json", "terminal"],
        default="json",
        help="Stdout format. JSON remains the default for backward compatibility.",
    )
    validate_parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return exit code 1 for warning-only validation results.",
    )
    _add_output_argument(validate_parser)

    gate_parser = subparsers.add_parser(
        "gate",
        help="Run contract and/or drift checks as one CI-friendly quality gate.",
    )
    gate_parser.add_argument("current", type=Path, help="Current dataset path.")
    gate_parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference dataset for drift checks.",
    )
    gate_parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Optional JSON data contract for exact validation.",
    )
    gate_parser.add_argument(
        "--columns",
        default=None,
        help="Optional comma-separated columns to include in drift checks.",
    )
    gate_parser.add_argument(
        "--max-columns",
        type=int,
        default=30,
        help="Maximum number of shared columns to compare for drift.",
    )
    gate_parser.add_argument(
        "--drift-warn-on",
        choices=["stable", "minor", "moderate", "severe"],
        default="moderate",
        help="Drift severity that changes a passing gate to warning.",
    )
    gate_parser.add_argument(
        "--drift-fail-on",
        choices=["stable", "minor", "moderate", "severe"],
        default="severe",
        help="Drift severity that fails the gate.",
    )
    gate_parser.add_argument(
        "--fail-on-validation-warning",
        action="store_true",
        help="Promote contract validation warnings to gate failures.",
    )
    gate_parser.add_argument(
        "--format",
        choices=["json", "terminal"],
        default="terminal",
        help="Stdout format.",
    )
    _add_output_argument(gate_parser)

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


def _write_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _emit(payload: dict, output: Path | None) -> None:
    if output is not None:
        _write_json(payload, output)
    print(json.dumps(payload, indent=2, default=str))


def _render_source(source: dict) -> str:
    rows = source.get("rows")
    columns = source.get("columns")
    size_bytes = source.get("size_bytes")
    lines = [
        "FrameVitals source",
        f"Name            {source.get('name', 'unknown')}",
        f"Kind            {source.get('kind', 'unknown')}",
        f"Format          {source.get('format', 'unknown')}",
        f"Rows            {rows if rows is not None else 'unknown'}",
        f"Columns         {columns if columns is not None else 'unknown'}",
        f"Size bytes      {size_bytes if size_bytes is not None else 'unknown'}",
        f"Materialized    {'yes' if source.get('materialized') else 'no'}",
        f"Projection      {'yes' if source.get('supports_projection') else 'no'}",
        f"Streaming       {'yes' if source.get('supports_streaming') else 'no'}",
    ]
    return "\n".join(lines)


def _render_snapshot(snapshot: dict) -> str:
    state = snapshot.get("state", {})
    source = snapshot.get("source", {})
    dataset = state.get("dataset", {})
    shape = dataset.get("shape", {})
    return "\n".join([
        "FrameVitals snapshot",
        f"File            {source.get('filename', 'unknown')}",
        f"Mode            {state.get('analysis_mode', 'unknown')}",
        f"Rows            {shape.get('rows', 'unknown')}",
        f"Columns         {shape.get('columns', 'unknown')}",
        f"Fingerprint     {snapshot.get('fingerprint', 'unknown')}",
    ])


def _render_snapshot_diff(report: dict) -> str:
    schema = report.get("schema", {})
    findings = report.get("findings", {})
    return "\n".join([
        "FrameVitals snapshot diff",
        f"Changed         {'yes' if report.get('changed') else 'no'}",
        f"Added columns   {len(schema.get('added_columns', []))}",
        f"Removed columns {len(schema.get('removed_columns', []))}",
        f"Type changes    {len(schema.get('type_changes', {}))}",
        f"New findings    {len(findings.get('new', []))}",
        f"Resolved        {len(findings.get('resolved', []))}",
    ])


def _render_system_info(info: dict) -> str:
    native = info.get("native", {})
    gpu = info.get("gpu", {})
    lines = [
        "FrameVitals system info",
        f"Python          {info.get('python', 'unknown')}",
        f"Backend         {info.get('backend', 'unknown')}",
        f"Native          available: {native.get('available', False)}",
    ]
    if gpu:
        lines.append(f"GPU             available: {gpu.get('available', False)}")
    return "\n".join(lines)


def _render_validation(report: dict) -> str:
    summary = report.get("summary", {})
    lines = [
        "FrameVitals validation",
        f"Status          {str(report.get('status', 'unknown')).upper()}",
        f"Columns checked {summary.get('columns_checked', 0)}",
        f"Errors          {summary.get('errors', 0)}",
        f"Warnings        {summary.get('warnings', 0)}",
    ]
    findings = report.get("findings", [])
    if findings:
        lines.append("")
        lines.append("Top findings")
        for finding in findings[:8]:
            lines.append(
                f"- [{str(finding.get('severity', 'error')).upper()}] "
                f"{finding.get('column')}: {finding.get('message')}"
            )
    return "\n".join(lines)


def _render_compare(report: dict) -> str:
    if not report.get("available"):
        return f"FrameVitals drift\nStatus          UNAVAILABLE\nReason          {report.get('reason')}"

    summary = report.get("summary", {})
    gate = report.get("gate", {})
    schema = report.get("schema", {})
    lines = [
        "FrameVitals drift",
        f"Gate            {str(gate.get('status', 'unknown')).upper()}",
        f"Severity        {str(summary.get('overall_verdict', 'unknown')).upper()}",
        f"Columns checked {summary.get('n_columns_compared', 0)}",
        f"Added columns   {len(schema.get('added_columns', []))}",
        f"Removed columns {len(schema.get('removed_columns', []))}",
        f"Type changes    {len(schema.get('dtype_changes', []))}",
    ]
    columns = report.get("columns", [])
    notable = [
        entry
        for entry in columns
        if entry.get("drift_severity") in {"minor", "moderate", "severe"}
    ]
    if notable:
        lines.append("")
        lines.append("Top drift")
        for entry in notable[:8]:
            lines.append(
                f"- [{str(entry.get('drift_severity')).upper()}] {entry.get('column')}"
            )
    return "\n".join(lines)


def _render_gate(report: dict) -> str:
    lines = [
        "FrameVitals gate",
        f"Status          {str(report.get('status', 'unknown')).upper()}",
        f"Passed          {'yes' if report.get('passed') else 'no'}",
        f"Checks          {', '.join(report.get('checks_run', [])) or 'none'}",
    ]
    reasons = report.get("reasons", [])
    if reasons:
        lines.append("")
        lines.append("Reasons")
        for reason in reasons[:10]:
            lines.append(f"- {reason}")
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        from framevitals.analysis_api import analyze

        report = analyze(
            args.file,
            target=args.target,
            mode=args.mode,
            artifacts=args.artifacts,
            workers=args.workers,
            preset=args.preset,
            config=args.config,
            disabled_modules=args.disabled_modules,
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
                disabled = resolved.get("disabled_modules") or ()
                if disabled:
                    print(f"Disabled      {', '.join(disabled)}")
            if args.output is not None:
                print(f"Full JSON     {args.output}")
            if args.html_report is not None:
                print(f"HTML report   {args.html_report}")
        return 0

    if args.command == "inspect":
        from framevitals.sources import inspect_source

        result = inspect_source(args.file)
        if args.output is not None:
            _write_json(result, args.output)
        if args.format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(_render_source(result))
            if args.output is not None:
                print(f"Source JSON    {args.output}")
        return 0

    if args.command == "snapshot":
        from framevitals.analysis_api import analyze

        report = analyze(
            args.file,
            target=args.target,
            mode=args.mode,
            artifacts=False,
            workers=args.workers,
            preset=args.preset,
            config=args.config,
            disabled_modules=args.disabled_modules,
        )
        snapshot = report.snapshot(args.output)
        if args.format == "json":
            print(snapshot.to_json())
        else:
            print(_render_snapshot(snapshot))
            if args.output is not None:
                print(f"Snapshot JSON   {args.output}")
        return 0

    if args.command == "compare-snapshots":
        from framevitals.snapshots import compare_snapshots, load_snapshot

        reference = load_snapshot(args.reference)
        current = load_snapshot(args.current)
        result = compare_snapshots(reference, current)
        if args.output is not None:
            _write_json(result, args.output)
        if args.format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(_render_snapshot_diff(result))
        if args.fail_on_change and result.get("changed"):
            return 1
        return 0

    if args.command == "system-info":
        from framevitals.acceleration import system_info

        result = system_info(probe_gpu=args.probe_gpu)
        if args.output is not None:
            _write_json(result, args.output)
        if args.format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(_render_system_info(result))
            if args.output is not None:
                print(f"System JSON     {args.output}")
        return 0

    if args.command == "plan":
        from framevitals.planning_api import plan

        result = plan(
            args.file,
            target=args.target,
            mode=args.mode,
            workers=args.workers,
            preset=args.preset,
            config=args.config,
            disabled_modules=args.disabled_modules,
        )
        if args.output is not None:
            _write_json(dict(result), args.output)
        if args.format == "json":
            print(json.dumps(dict(result), indent=2, default=str))
        else:
            print(result.explain_text())
            disabled = result.get("config", {}).get("disabled_modules") or ()
            if disabled:
                print(f"Disabled modules: {', '.join(disabled)}")
            if args.output is not None:
                print(f"Plan JSON     {args.output}")
        return 0

    if args.command == "clean":
        from framevitals.operations import clean, plan_cleaning
        from framevitals.security import sanitize_csv_value

        cleaning_plan = plan_cleaning(args.file)
        if args.plan_output is not None:
            cleaning_plan.to_json(args.plan_output)

        payload = {
            "plan": dict(cleaning_plan),
            "summary": cleaning_plan.summary(),
            "cleaned_output": str(args.output) if args.output is not None else None,
        }

        if args.output is not None:
            if args.output.suffix.lower() != ".csv":
                raise ValueError("The clean command currently writes CSV output only.")
            cleaned = clean(args.file, plan=cleaning_plan)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cleaned.map(sanitize_csv_value).to_csv(args.output, index=False)

        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.command == "compare":
        from framevitals.drift_analysis import severity_at_least
        from framevitals.operations import compare

        report = compare(
            args.reference,
            args.current,
            columns=_parse_columns(args.columns),
            max_columns=args.max_columns,
        )
        if args.output is not None:
            _write_json(report, args.output)
        if args.format == "terminal":
            print(_render_compare(report))
        else:
            print(json.dumps(report, indent=2, default=str))

        if args.fail_on and severity_at_least(
            report.get("gate", {}).get("severity", "unknown"),
            args.fail_on,
        ):
            return 1
        return 0

    if args.command == "infer-contract":
        from framevitals.operations import infer_contract

        report = infer_contract(
            args.file,
            numeric_tolerance=args.numeric_tolerance,
            max_categories=args.max_categories,
            null_fraction_tolerance=args.null_fraction_tolerance,
            infer_unique=args.infer_unique,
            min_unique_rows=args.min_unique_rows,
            allow_extra_columns=args.allow_extra_columns,
        )
        _emit(report, args.output)
        return 0

    if args.command == "validate":
        from framevitals.operations import validate

        report = validate(
            args.file,
            _load_contract(args.contract),
        )
        if args.output is not None:
            _write_json(report, args.output)
        if args.format == "terminal":
            print(_render_validation(report))
        else:
            print(json.dumps(report, indent=2, default=str))

        if report.get("status") == "fail":
            return 2
        if report.get("status") == "warn" and args.fail_on_warn:
            return 1
        return 0

    if args.command == "gate":
        from framevitals.operations import gate

        contract = _load_contract(args.contract) if args.contract is not None else None
        report = gate(
            args.current,
            reference=args.reference,
            contract=contract,
            columns=_parse_columns(args.columns),
            max_columns=args.max_columns,
            drift_warn_on=args.drift_warn_on,
            drift_fail_on=args.drift_fail_on,
            fail_on_validation_warning=args.fail_on_validation_warning,
        )
        if args.output is not None:
            _write_json(report, args.output)
        if args.format == "json":
            print(json.dumps(report, indent=2, default=str))
        else:
            print(_render_gate(report))
        return 0 if report.get("passed") else 1

    if args.command == "config":
        from framevitals.config import resolve_config

        resolved = resolve_config(args.file, preset=args.preset)
        print(json.dumps(resolved.to_dict(), indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
