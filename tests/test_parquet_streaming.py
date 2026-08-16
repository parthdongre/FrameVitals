import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import framevitals
from framevitals.sources import ParquetSource, resolve_source


def _write_parquet(path, rows: int = 10_000) -> pd.DataFrame:
    frame = pd.DataFrame({
        "value": np.arange(rows, dtype=np.float64),
        "other": np.arange(rows, dtype=np.float64) * 2.0 + 3.0,
        "group": [f"g-{index % 7}" for index in range(rows)],
        "event_time": pd.date_range("2026-01-01", periods=rows, freq="min"),
    })
    frame.loc[::101, "value"] = np.nan
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, row_group_size=777)
    return frame


def test_parquet_source_exposes_metadata_projection_and_batches(tmp_path):
    path = tmp_path / "dataset.parquet"
    frame = _write_parquet(path, rows=2_500)

    source = resolve_source(path)
    assert isinstance(source, ParquetSource)

    metadata = source.inspect()
    assert metadata.rows == len(frame)
    assert metadata.columns == len(frame.columns)
    assert metadata.format == "parquet"
    assert metadata.materialized is False
    assert metadata.supports_projection is True
    assert metadata.supports_streaming is True

    batches = list(source.iter_batches(batch_size=600, columns=["value", "group"]))
    assert sum(batch.num_rows for batch in batches) == len(frame)
    assert batches[0].schema.names == ["value", "group"]
    assert max(batch.num_rows for batch in batches) <= 600


def test_public_profile_streams_parquet_without_calling_load(tmp_path, monkeypatch):
    path = tmp_path / "stream.parquet"
    frame = _write_parquet(path, rows=12_000)

    def fail_load(self):
        raise AssertionError("streaming profile must not materialize the complete Parquet file")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.profile(path)

    assert result["dataset_name"] == "stream.parquet"
    assert result["shape"] == {"rows": len(frame), "columns": 4}
    assert result["streaming_metadata"]["enabled"] is True
    assert result["streaming_metadata"]["full_materialization"] is False
    assert result["streaming_metadata"]["sample_rows"] == len(frame)
    assert result["source_metadata"]["supports_streaming"] is True
    assert result["missing_counts"]["value"] == int(frame["value"].isna().sum())
    assert result["numeric_summary"]["value"]["count"] == int(frame["value"].notna().sum())
    assert result["numeric_summary"]["other"]["mean"] == pytest.approx(
        round(float(frame["other"].mean()), 3)
    )
    assert "group" in result["categorical_summary"]
    assert "event_time" in result["date_columns"]


def test_large_parquet_retains_only_bounded_row_sample(tmp_path):
    path = tmp_path / "large.parquet"
    frame = _write_parquet(path, rows=60_000)

    result = framevitals.profile(path)

    streaming = result["streaming_metadata"]
    assert streaming["full_materialization"] is False
    assert streaming["sample_rows"] == 50_000
    assert streaming["sample_strategy"] == "evenly_spaced_global_rows"
    assert result["categorical_summary_metadata"]["sampled"] is True
    assert result["correlation_metadata"]["row_sampled"] is True
    assert result["duplicate_metadata"]["sampled"] is True
    assert result["missing_counts"]["value"] == int(frame["value"].isna().sum())
