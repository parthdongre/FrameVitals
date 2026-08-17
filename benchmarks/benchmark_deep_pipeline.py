"""Reproducible 8k x 180 Deep pipeline benchmark.

This workload is the stable performance target for the August optimization
track.  It is intentionally numeric and wide so profiling, adaptive deep
statistics, anomaly screening, and pairwise diagnostics all receive work.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from framevitals.pipeline import run_full_analysis


DEFAULT_ROWS = 8_000
DEFAULT_COLUMNS = 180
HISTORICAL_ORIGINAL_SECONDS = 76.17
TEN_X_TARGET_SECONDS = HISTORICAL_ORIGINAL_SECONDS / 10.0


def make_workload(
    rows: int = DEFAULT_ROWS,
    columns: int = DEFAULT_COLUMNS,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    if rows < 100 or columns < 32:
        raise ValueError("benchmark requires at least 100 rows and 32 columns")

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, columns))

    # Make a bounded set of columns diagnostically interesting so deep triage
    # consistently exercises skew, missingness, constants, and relationships.
    skewed = min(12, columns)
    values[:, :skewed] = rng.lognormal(
        mean=0.0,
        sigma=1.5,
        size=(rows, skewed),
    )

    for index in range(12, min(24, columns)):
        values[::11, index] = np.nan

    for index in range(24, min(28, columns)):
        values[:, index] = float(index)

    for offset, index in enumerate(range(30, min(40, columns))):
        source = offset % min(10, columns)
        values[:, index] = values[:, source] * 0.97 + rng.normal(
            scale=0.03,
            size=rows,
        )

    return pd.DataFrame(
        values,
        columns=[f"n{index:03d}" for index in range(columns)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("FRAMEVITALS_BACKEND", "numpy")
    frame = make_workload(args.rows, args.columns)

    started = time.perf_counter()
    result = run_full_analysis(
        "benchmark-8k-180-deep",
        original_filename="synthetic-8k-180",
        analysis_mode="deep",
        skip_ai=True,
        parallel_workers=4,
        dataframe=frame,
        write_artifacts=False,
    )
    elapsed = time.perf_counter() - started

    payload = {
        "benchmark_schema_version": 1,
        "workload": {
            "rows": int(args.rows),
            "columns": int(args.columns),
            "kind": "deterministic_wide_numeric_deep",
            "seed": 42,
            "backend": os.environ.get("FRAMEVITALS_BACKEND", "auto"),
        },
        "elapsed_seconds": round(float(elapsed), 6),
        "historical_original_seconds": HISTORICAL_ORIGINAL_SECONDS,
        "ten_x_target_seconds": round(TEN_X_TARGET_SECONDS, 6),
        "speedup_vs_historical_original": round(
            HISTORICAL_ORIGINAL_SECONDS / max(elapsed, 1.0e-12),
            4,
        ),
        "ten_x_target_met": bool(elapsed <= TEN_X_TARGET_SECONDS),
        "pipeline_timings_ms": result.get("timings_ms", {}),
        "deep_execution": (
            result.get("deep_statistics_v2", {}).get("execution", {})
            if isinstance(result.get("deep_statistics_v2"), dict)
            else {}
        ),
    }

    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
