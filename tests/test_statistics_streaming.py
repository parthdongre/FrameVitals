import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import framevitals
from framevitals.sources import ParquetSource


def _write_statistics_parquet(path, rows: int = 12_000) -> pd.DataFrame:
    frame = pd.DataFrame({
        "value": np.linspace(1.0, 100.0, rows),
        "other": np.linspace(3.0, 203.0, rows),
        "group": [f"g-{index % 5}" for index in range(rows)],
        "event_time": pd.date_range("2026-01-01", periods=rows, freq="min"),
    })
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        row_group_size=777,
    )
    return frame


def test_public_statistics_streams_bounded_parquet_sample(tmp_path, monkeypatch):
    path = tmp_path / "statistics.parquet"
    frame = _write_statistics_parquet(path)

    def fail_load(self):
        raise AssertionError("statistics must not materialize the complete Parquet file")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.statistics(path, mode="quick", max_pairs=2)

    assert result["dataset_name"] == "statistics.parquet"
    execution = result["execution"]
    assert execution["execution_schema_version"] == "1"
    assert execution["method"] == "bounded_deep_statistics"
    assert execution["scope"] == "bounded_deep_statistics"
    assert execution["full_materialization"] is False
    assert execution["source_rows"] == len(frame)
    assert execution["source_columns"] == len(frame.columns)
    assert execution["sample_rows"] == 1_000
    assert execution["sampled"] is True
    assert execution["strategy"] == "streaming_stratified_jitter_global_rows"
    assert execution["pair_budget"] == 2
    assert execution["source"]["format"] == "parquet"
    assert result["source"]["supports_streaming"] is True
