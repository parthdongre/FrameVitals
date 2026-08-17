"""Generate and benchmark a real 100,000 x 10,000 Parquet dataset.

The dataset is written in bounded row groups, so generation never materializes
all one billion cells in RAM. The on-disk Parquet file is then analyzed through
the public ``framevitals.analyze`` API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import framevitals as fv


DEFAULT_ROWS = 100_000
DEFAULT_COLUMNS = 10_000
PATTERN_COUNT = 32
VALID_MODES = ("quick", "standard", "deep", "research")


def _pattern_numpy(pattern_id: int, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return one deterministic int16 pattern and optional null mask."""
    pattern_id = int(pattern_id) % PATTERN_COUNT
    if pattern_id == 0:
        values = row_ids % 101
    elif pattern_id == 1:
        values = 2 * (row_ids % 101)
    elif pattern_id == 2:
        values = (row_ids * 7 + 3) % 211
    elif pattern_id == 3:
        values = np.full(row_ids.shape, 7, dtype=np.int64)
    elif pattern_id == 4:
        values = (row_ids * 13) % 503
        values = values.copy()
        values[row_ids % 997 == 0] = 30_000
    elif pattern_id == 5:
        values = (row_ids * 5) % 307
        return values.astype(np.int16, copy=False), (row_ids % 10 == 0)
    elif pattern_id == 6:
        values = row_ids % 2
    elif pattern_id == 7:
        values = np.where(row_ids % 100 < 50, 10, 200)
    else:
        multiplier = pattern_id * 2 + 1
        values = (row_ids * multiplier + pattern_id * 17) % 997
    return values.astype(np.int16, copy=False), None


def _pattern_arrow(pattern_id: int, row_ids: np.ndarray) -> pa.Array:
    values, mask = _pattern_numpy(pattern_id, row_ids)
    return pa.array(values, mask=mask, type=pa.int16())


def generate_dataset(
    path: Path,
    *,
    rows: int = DEFAULT_ROWS,
    columns: int = DEFAULT_COLUMNS,
    row_group_rows: int = 10_000,
) -> dict[str, Any]:
    """Write the full logical dataset to Parquet using bounded-memory row groups."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"n{index:05d}" for index in range(columns)]
    schema = pa.schema([pa.field(name, pa.int16()) for name in names])

    started = time.perf_counter()
    writer = pq.ParquetWriter(
        path,
        schema=schema,
        compression="snappy",
        use_dictionary=True,
        write_statistics=False,
    )
    try:
        for start in range(0, rows, row_group_rows):
            stop = min(rows, start + row_group_rows)
            row_ids = np.arange(start, stop, dtype=np.int64)
            patterns = [_pattern_arrow(index, row_ids) for index in range(PATTERN_COUNT)]
            arrays = [patterns[index % PATTERN_COUNT] for index in range(columns)]
            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            writer.write_table(pa.Table.from_batches([batch], schema=schema), row_group_size=len(row_ids))
    finally:
        writer.close()

    elapsed = time.perf_counter() - started
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path),
        "rows": int(parquet.metadata.num_rows),
        "columns": int(parquet.metadata.num_columns),
        "cells": int(parquet.metadata.num_rows) * int(parquet.metadata.num_columns),
        "file_size_mb": round(path.stat().st_size / 1_000_000, 3),
        "dense_int16_raw_gb": round(rows * columns * 2 / 1_000_000_000, 3),
        "generation_seconds": round(elapsed, 6),
        "row_groups": int(parquet.metadata.num_row_groups),
        "pattern_count": PATTERN_COUNT,
    }


def _expected_metrics(rows: int) -> dict[int, dict[str, Any]]:
    row_ids = np.arange(rows, dtype=np.int64)
    expected: dict[int, dict[str, Any]] = {}
    for pattern_id in range(PATTERN_COUNT):
        values, mask = _pattern_numpy(pattern_id, row_ids)
        valid = values if mask is None else values[~mask]
        expected[pattern_id] = {
            "missing": 0 if mask is None else int(mask.sum()),
            "count": int(valid.size),
            "mean": float(valid.astype(np.float64).mean()) if valid.size else None,
            "min": int(valid.min()) if valid.size else None,
            "max": int(valid.max()) if valid.size else None,
        }
    return expected


def _accuracy_checks(result: dict[str, Any], *, rows: int) -> dict[str, Any]:
    profile = result.get("profile", {})
    columns = list(profile.get("columns", []))
    missing_counts = profile.get("missing_counts", {})
    numeric_summary = profile.get("numeric_summary", {})
    expected = _expected_metrics(rows)

    missing_matches = 0
    count_matches = 0
    minmax_matches = 0
    mean_errors: list[float] = []
    expected_missing_total = 0

    for column in columns:
        index = int(str(column)[1:])
        truth = expected[index % PATTERN_COUNT]
        expected_missing_total += int(truth["missing"])
        if int(missing_counts.get(column, -1)) == int(truth["missing"]):
            missing_matches += 1

        summary = numeric_summary.get(column, {})
        if int(summary.get("count", -1)) == int(truth["count"]):
            count_matches += 1
        if summary.get("min") == truth["min"] and summary.get("max") == truth["max"]:
            minmax_matches += 1
        observed_mean = summary.get("mean")
        if observed_mean is not None and truth["mean"] is not None:
            mean_errors.append(abs(float(observed_mean) - float(truth["mean"])))

    expected_health_missing = (
        expected_missing_total / max(rows * len(columns), 1) * 100.0
    )
    observed_health_missing = float(
        result.get("health", {}).get("details", {}).get("missing_percent", 0.0) or 0.0
    )

    deep_stats = result.get("deep_statistics_v2", {})
    numeric_stats = deep_stats.get("numeric_statistics", {}) if isinstance(deep_stats, dict) else {}
    deep_mean_errors: list[float] = []
    for column, summary in numeric_stats.items():
        if not isinstance(summary, dict) or summary.get("mean") is None:
            continue
        index = int(str(column)[1:])
        truth_mean = expected[index % PATTERN_COUNT]["mean"]
        if truth_mean is not None:
            deep_mean_errors.append(abs(float(summary["mean"]) - float(truth_mean)))

    timings = result.get("timings_ms", {})
    phase3 = timings.get("phase3_tasks", {}) if isinstance(timings, dict) else {}
    return {
        "profiled_columns_checked": len(columns),
        "missing_count_exact_matches": missing_matches,
        "numeric_count_exact_matches": count_matches,
        "numeric_minmax_exact_matches": minmax_matches,
        "profile_mean_max_abs_error": round(max(mean_errors), 6) if mean_errors else None,
        "profile_mean_mean_abs_error": round(float(np.mean(mean_errors)), 6) if mean_errors else None,
        "expected_projected_missing_percent": round(expected_health_missing, 6),
        "health_missing_percent": round(observed_health_missing, 6),
        "health_missing_abs_error": round(abs(observed_health_missing - expected_health_missing), 6),
        "deep_numeric_columns_checked": len(numeric_stats),
        "deep_mean_max_abs_error": round(max(deep_mean_errors), 6) if deep_mean_errors else None,
        "deep_mean_mean_abs_error": round(float(np.mean(deep_mean_errors)), 6) if deep_mean_errors else None,
        "phase3_tasks_present": sorted(phase3) if isinstance(phase3, dict) else [],
        "full_pipeline_sections_present": sorted(
            key
            for key in (
                "profile",
                "column_roles",
                "health",
                "ml_readiness",
                "quality_diagnostics",
                "advanced",
                "deep_statistics_v2",
                "anomalies_v2",
                "time_series",
            )
            if key in result
        ),
    }


def benchmark_dataset(path: Path, *, mode: str) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    rows = int(parquet.metadata.num_rows)
    columns = int(parquet.metadata.num_columns)

    started = time.perf_counter()
    result = fv.analyze(path, mode=mode, artifacts=False, workers=4)
    analysis_seconds = time.perf_counter() - started

    streaming = result.get("execution", {}).get("streaming", {})
    return {
        "benchmark_schema_version": 1,
        "workload": {
            "rows": rows,
            "columns": columns,
            "cells": rows * columns,
            "file_size_mb": round(path.stat().st_size / 1_000_000, 3),
            "dense_int16_raw_gb": round(rows * columns * 2 / 1_000_000_000, 3),
            "mode": mode,
            "source": "real_parquet_file",
        },
        "analysis_seconds": round(analysis_seconds, 6),
        "pipeline_timings_ms": result.get("timings_ms", {}),
        "streaming_execution": streaming,
        "accuracy": _accuracy_checks(result, rows=rows),
        "safety": {
            "full_materialization": streaming.get("full_materialization"),
            "source_rows": streaming.get("source_rows"),
            "source_columns": streaming.get("source_columns"),
            "profiled_columns": streaming.get("profiled_columns"),
            "working_sample_rows": streaming.get("working_sample_rows"),
            "column_sampled": streaming.get("column_sampled"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("wide-100k-x-10k.parquet"))
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--row-group-rows", type=int, default=10_000)
    parser.add_argument("--mode", choices=VALID_MODES)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.generate_only:
        payload = generate_dataset(
            args.dataset,
            rows=args.rows,
            columns=args.columns,
            row_group_rows=args.row_group_rows,
        )
    else:
        if not args.dataset.exists():
            generate_dataset(
                args.dataset,
                rows=args.rows,
                columns=args.columns,
                row_group_rows=args.row_group_rows,
            )
        if args.mode is None:
            parser.error("--mode is required unless --generate-only is used")
        payload = benchmark_dataset(args.dataset, mode=args.mode)

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
