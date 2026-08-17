from types import SimpleNamespace

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

import framevitals as fv
from framevitals.execution import derive_streaming_profile_column_limit
from framevitals.sources import DatasetMetadata


class _VirtualSchema:
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


class _VirtualWideSource:
    def __init__(self, *, rows: int, columns: int):
        self.rows = int(rows)
        self.columns = int(columns)
        self.max_requested_columns = 0
        self.unbounded_requested = False
        self._schema = _VirtualSchema(columns)

    def inspect(self):
        return DatasetMetadata(
            name="virtual-wide",
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
        return self._schema

    def iter_batches(self, *, batch_size=65_536, columns=None):
        if columns is None:
            self.unbounded_requested = True
            raise AssertionError("ultra-wide source must always be projected")
        names = list(columns)
        self.max_requested_columns = max(self.max_requested_columns, len(names))
        rows = min(int(batch_size), self.rows)
        values = pa.array(np.arange(rows, dtype=np.float64))
        yield pa.RecordBatch.from_arrays([values] * len(names), names=names)

    def load(self):
        raise AssertionError("virtual wide source must never materialize")


def test_extreme_deep_streaming_profile_column_limit_is_128():
    assert (
        derive_streaming_profile_column_limit(
            1_000_000,
            100_000,
            mode="deep",
        )
        == 128
    )


def test_extreme_plan_projects_100k_column_schema():
    source = _VirtualWideSource(rows=1_000_000, columns=100_000)

    result = fv.plan(source, mode="deep")

    assert result["shape"] == {"rows": 1_000_000, "columns": 100_000}
    assert result["execution_budget"]["scale_class"] == "extreme"
    assert result["planning_data"]["column_sampled"] is True
    assert result["planning_data"]["sample_columns"] == 128
    assert result["planning_data"]["source_columns"] == 100_000
    assert source.unbounded_requested is False
    assert source.max_requested_columns == 128


def test_ultra_wide_streaming_analysis_never_requests_full_width():
    source = _VirtualWideSource(rows=1_000, columns=10_000)

    result = fv.analyze(source, mode="deep", artifacts=False, workers=2)

    streaming = result["execution"]["streaming"]
    assert streaming["source_rows"] == 1_000
    assert streaming["source_columns"] == 10_000
    assert streaming["profiled_columns"] == 128
    assert streaming["column_sampled"] is True
    assert streaming["column_strategy"] == "deterministic_schema_projection"
    assert result["profile"]["shape"] == {"rows": 1_000, "columns": 10_000}
    assert result["health"]["execution"]["profiled_columns"] == 128
    assert source.unbounded_requested is False
    assert source.max_requested_columns == 128
