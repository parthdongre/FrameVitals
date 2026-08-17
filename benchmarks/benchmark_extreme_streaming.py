"""Virtual ultra-wide streaming benchmark.

A dense materialization of the target shapes is intentionally avoided. The
benchmark models a projection-capable streaming source and fails if FrameVitals
ever requests the complete ultra-wide schema during analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

import framevitals as fv
from framevitals.sources import DatasetMetadata


DEFAULT_ROWS = 1_000_000
DEFAULT_COLUMNS = 100_000
VALID_MODES = ("quick", "standard", "deep", "research")


class VirtualSchema:
    def __init__(self, columns: int):
        self.columns = int(columns)
        self.dtype = pa.float64()

    def __iter__(self):
        for index in range(self.columns):
            yield SimpleNamespace(name=f"n{index:05d}", type=self.dtype)

    def field(self, name: str):
        index = int(name[1:])
        if index < 0 or index >= self.columns:
            raise KeyError(name)
        return SimpleNamespace(name=name, type=self.dtype)


class VirtualExtremeSource:
    def __init__(self, rows: int, columns: int):
        self.rows = int(rows)
        self.columns = int(columns)
        self.schema_view = VirtualSchema(columns)
        self.max_requested_columns = 0
        self.unbounded_requests = 0
        self.batches_yielded = 0
        self.rows_yielded = 0

    def inspect(self):
        return DatasetMetadata(
            name=f"virtual-{self.rows}-x-{self.columns}",
            kind="virtual",
            format="synthetic",
            rows=self.rows,
            columns=self.columns,
            size_bytes=self.rows * self.columns * 8,
            materialized=False,
            supports_projection=True,
            supports_streaming=True,
        )

    def schema(self):
        return self.schema_view

    def iter_batches(self, *, batch_size=65_536, columns=None):
        if columns is None:
            self.unbounded_requests += 1
            raise RuntimeError(
                f"FrameVitals attempted an unbounded {self.columns:,}-column scan."
            )

        names = list(columns)
        self.max_requested_columns = max(self.max_requested_columns, len(names))
        offset = 0
        while offset < self.rows:
            take = min(int(batch_size), self.rows - offset)
            base = np.arange(offset, offset + take, dtype=np.float64)
            values = pa.array((base % 10_007) / 10_007.0)
            batch = pa.RecordBatch.from_arrays([values] * len(names), names=names)
            self.batches_yielded += 1
            self.rows_yielded += take
            yield batch
            offset += take

    def load(self):
        raise RuntimeError("Virtual extreme source must never materialize fully.")

    def reset_observations(self) -> None:
        self.max_requested_columns = 0
        self.unbounded_requests = 0
        self.batches_yielded = 0
        self.rows_yielded = 0


def run_benchmark(*, rows: int, columns: int, mode: str) -> dict:
    source = VirtualExtremeSource(rows, columns)

    plan_started = time.perf_counter()
    plan = fv.plan(source, mode=mode, workers=4)
    plan_seconds = time.perf_counter() - plan_started

    source.reset_observations()

    analysis_started = time.perf_counter()
    result = fv.analyze(
        source,
        mode=mode,
        artifacts=False,
        workers=4,
    )
    analysis_seconds = time.perf_counter() - analysis_started

    streaming = result.get("execution", {}).get("streaming", {})
    working_sample_rows = int(streaming.get("working_sample_rows", 0) or 0)
    return {
        "benchmark_schema_version": 2,
        "workload": {
            "rows": rows,
            "columns": columns,
            "cells": rows * columns,
            "dense_float64_raw_gb": round(rows * columns * 8 / 1_000_000_000, 3),
            "mode": mode,
            "source": "virtual_projection_capable_stream",
        },
        "plan_seconds": round(plan_seconds, 6),
        "analysis_seconds": round(analysis_seconds, 6),
        "scale_class": plan.get("execution_budget", {}).get("scale_class"),
        "planning": plan.get("planning_data", {}),
        "streaming_execution": streaming,
        "pipeline_timings_ms": result.get("timings_ms", {}),
        "source_observed": {
            "max_requested_columns": source.max_requested_columns,
            "unbounded_requests": source.unbounded_requests,
            "batches_yielded": source.batches_yielded,
            "rows_yielded": source.rows_yielded,
        },
        "safety": {
            "full_materialization": streaming.get("full_materialization"),
            "column_sampled": streaming.get("column_sampled"),
            "profiled_columns": streaming.get("profiled_columns"),
            "source_columns": streaming.get("source_columns"),
            "working_sample_rows": working_sample_rows,
            "all_source_rows_scanned": source.rows_yielded == rows,
            "unbounded_width_requested": source.unbounded_requests > 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--columns", type=int, default=DEFAULT_COLUMNS)
    parser.add_argument("--mode", choices=VALID_MODES, default="deep")
    parser.add_argument("--output", type=Path, default=Path("extreme-benchmark-results.json"))
    args = parser.parse_args()

    payload = run_benchmark(rows=args.rows, columns=args.columns, mode=args.mode)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
