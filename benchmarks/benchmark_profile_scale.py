#!/usr/bin/env python3
"""Reproducible FrameVitals scale benchmark.

The harness runs each scenario in a fresh subprocess so process peak RSS values
are comparable and one scenario's allocator state cannot contaminate another.
It intentionally records measurements rather than enforcing fragile absolute
performance thresholds in CI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROWS = 100_000
DEFAULT_NUMERIC_COLUMNS = 100
DEFAULT_CATEGORICAL_COLUMNS = 5
DEFAULT_SEED = 42
SCENARIOS = ("numpy", "auto", "parquet")


def _peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    if platform.system() == "Darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _frame(
    rows: int,
    numeric_columns: int,
    categorical_columns: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, numeric_columns), dtype=np.float64)
    if numeric_columns:
        values[::97, ::7] = np.nan
    frame = pd.DataFrame(
        values,
        columns=[f"metric_{index:03d}" for index in range(numeric_columns)],
        copy=False,
    )
    for index in range(categorical_columns):
        cardinality = 7 + index * 3
        frame[f"category_{index:02d}"] = np.asarray(
            [f"g{index}_{row % cardinality}" for row in range(rows)],
            dtype=object,
        )
    return frame


def _write_parquet_streaming(
    path: Path,
    *,
    rows: int,
    numeric_columns: int,
    categorical_columns: int,
    seed: int,
    chunk_rows: int = 20_000,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            'Parquet benchmark requires: pip install "framevitals[arrow]"'
        ) from exc

    writer = None
    try:
        offset = 0
        chunk_index = 0
        while offset < rows:
            current = min(chunk_rows, rows - offset)
            frame = _frame(
                current,
                numeric_columns,
                categorical_columns,
                seed + chunk_index,
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table, row_group_size=min(current, 10_000))
            offset += current
            chunk_index += 1
    finally:
        if writer is not None:
            writer.close()


def _result_metadata(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": result.get("shape"),
        "numeric_summary_metadata": result.get("numeric_summary_metadata"),
        "correlation_metadata": result.get("correlation_metadata"),
        "duplicate_metadata": result.get("duplicate_metadata"),
        "streaming_metadata": result.get("streaming_metadata"),
        "source_metadata": result.get("source_metadata"),
    }


def _worker(args: argparse.Namespace) -> int:
    import framevitals
    from framevitals.backends import backend_status

    started = time.perf_counter()
    source_memory_mb = None

    if args.scenario in {"numpy", "auto"}:
        if args.scenario == "numpy":
            os.environ["FRAMEVITALS_BACKEND"] = "numpy"
        else:
            os.environ.pop("FRAMEVITALS_BACKEND", None)
        frame = _frame(
            args.rows,
            args.numeric_columns,
            args.categorical_columns,
            args.seed,
        )
        source_memory_mb = round(
            float(frame.memory_usage(index=True, deep=True).sum()) / (1024 * 1024),
            3,
        )
        result = framevitals.profile(frame)
    elif args.scenario == "parquet":
        if not args.parquet_path:
            raise ValueError("parquet worker requires --parquet-path")
        os.environ.pop("FRAMEVITALS_BACKEND", None)
        result = framevitals.profile(Path(args.parquet_path))
    else:
        raise ValueError(f"Unknown benchmark scenario: {args.scenario}")

    elapsed = time.perf_counter() - started
    payload = {
        "scenario": args.scenario,
        "rows": args.rows,
        "numeric_columns": args.numeric_columns,
        "categorical_columns": args.categorical_columns,
        "total_columns": args.numeric_columns + args.categorical_columns,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 6),
        "peak_rss_mb": round(_peak_rss_mb(), 3),
        "source_memory_mb": source_memory_mb,
        "backend_status": backend_status(),
        "result": _result_metadata(result),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _run_scenario(
    script: Path,
    scenario: str,
    *,
    rows: int,
    numeric_columns: int,
    categorical_columns: int,
    seed: int,
    parquet_path: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--scenario",
        scenario,
        "--rows",
        str(rows),
        "--numeric-columns",
        str(numeric_columns),
        "--categorical-columns",
        str(categorical_columns),
        "--seed",
        str(seed),
    ]
    if parquet_path is not None:
        command.extend(["--parquet-path", str(parquet_path)])

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Benchmark worker {scenario} produced no JSON output.")
    return json.loads(lines[-1])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--numeric-columns", type=int, default=DEFAULT_NUMERIC_COLUMNS)
    parser.add_argument(
        "--categorical-columns",
        type=int,
        default=DEFAULT_CATEGORICAL_COLUMNS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=SCENARIOS, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parquet-path", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if args.numeric_columns < 0 or args.categorical_columns < 0:
        parser.error("column counts must be non-negative")
    if args.numeric_columns + args.categorical_columns < 1:
        parser.error("at least one column is required")
    if args.worker and args.scenario is None:
        parser.error("--worker requires --scenario")
    return args


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _worker(args)

    script = Path(__file__).resolve()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="framevitals-benchmark-") as directory:
        parquet_path: Path | None = None
        if "parquet" in args.scenarios:
            parquet_path = Path(directory) / "scale.parquet"
            prepare_started = time.perf_counter()
            _write_parquet_streaming(
                parquet_path,
                rows=args.rows,
                numeric_columns=args.numeric_columns,
                categorical_columns=args.categorical_columns,
                seed=args.seed,
            )
            parquet_prepare_seconds = round(time.perf_counter() - prepare_started, 6)
        else:
            parquet_prepare_seconds = None

        for scenario in args.scenarios:
            results.append(
                _run_scenario(
                    script,
                    scenario,
                    rows=args.rows,
                    numeric_columns=args.numeric_columns,
                    categorical_columns=args.categorical_columns,
                    seed=args.seed,
                    parquet_path=parquet_path if scenario == "parquet" else None,
                )
            )

    payload = {
        "benchmark_schema_version": 1,
        "workload": {
            "rows": args.rows,
            "numeric_columns": args.numeric_columns,
            "categorical_columns": args.categorical_columns,
            "total_columns": args.numeric_columns + args.categorical_columns,
            "seed": args.seed,
        },
        "parquet_prepare_seconds": parquet_prepare_seconds,
        "measurements": results,
        "notes": [
            "Each measurement runs in a fresh process.",
            "Peak RSS includes imports and in-memory source construction for DataFrame scenarios.",
            "Parquet file preparation occurs outside the measured Parquet worker.",
            "No absolute timing threshold is enforced; compare like-for-like machines and commits.",
        ],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
