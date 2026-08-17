"""Virtual 1,000,000 x 100,000 Deep streaming benchmark.

A dense float64 materialization of this shape would require roughly 800 GB just
for raw cells, so the benchmark models a projection-capable streaming source.
FrameVitals must preserve the true source shape while never requesting the full
100,000-column width from the source.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pyarrow as pa

import framevitals as fv
from framevitals.sources import DatasetMetadata


ROWS = 1_000_000
COLUMNS = 100_000


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
    def __init__(self, rows: int = ROWS, columns: int = COLUMNS):
        self.rows = int(rows)
        self.columns = int(columns)
        self.schema_view = VirtualSchema(columns)
        self.max_requested_columns = 0
        self.unbounded_requests = 0
        self.batches_yielded = 0
        self.rows_yielded = 0

    def inspect(self):
        return DatasetMetadata(
            name="virtual-1m-x-100k",
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
                "FrameVitals attempted an unbounded 100,000-column scan."
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


def main() -> None:
    source = VirtualExtremeSource()

    plan_started = time.perf_counter()
    plan = fv.plan(source, mode="deep", workers=4)
    plan_seconds = time.perf_counter() - plan_started

    # Reset counters so execution accounting is independent of planning.
    source.max_requested_columns = 0
    source.unbounded_requests = 0
    source.batches_yielded = 0
    source.rows_yielded = 0

    analysis_started = time.perf_counter()
    result = fv.analyze(
        source,
        mode="deep",
        artifacts=False,
        workers=4,
    )
    analysis_seconds = time.perf_counter() - analysis_started

    streaming = result.get("execution", {}).get("streaming", {})
    payload = {
        "benchmark_schema_version": 1,
        "workload": {
            "rows": ROWS,
            "columns": COLUMNS,
            "cells": ROWS * COLUMNS,
            "dense_float64_raw_gb": round(ROWS * COLUMNS * 8 / 1_000_000_000, 3),
            "mode": "deep",
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
            "all_source_rows_scanned": source.rows_yielded == ROWS,
            "unbounded_width_requested": source.unbounded_requests > 0,
        },
    }

    output = Path("extreme-benchmark-results.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
