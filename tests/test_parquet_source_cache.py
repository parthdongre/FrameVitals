import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from framevitals.sources import ParquetSource


def test_parquet_source_reuses_one_parquet_file_handle(tmp_path, monkeypatch):
    path = tmp_path / "cache.parquet"
    frame = pd.DataFrame({
        "x": range(100),
        "y": [value * 2 for value in range(100)],
    })
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=17)

    real_parquet_file = pq.ParquetFile
    calls = {"count": 0}

    def counted_parquet_file(*args, **kwargs):
        calls["count"] += 1
        return real_parquet_file(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", counted_parquet_file)

    source = ParquetSource(path)
    first_metadata = source.inspect()
    second_metadata = source.inspect()
    first_schema = source.schema()
    second_schema = source.schema()
    first_batches = list(source.iter_batches(batch_size=23, columns=["x"]))
    second_batches = list(source.iter_batches(batch_size=31, columns=["y"]))

    assert calls["count"] == 1
    assert first_metadata is second_metadata
    assert first_schema is second_schema
    assert first_metadata.rows == 100
    assert first_metadata.columns == 2
    assert sum(batch.num_rows for batch in first_batches) == 100
    assert sum(batch.num_rows for batch in second_batches) == 100
    assert first_batches[0].schema.names == ["x"]
    assert second_batches[0].schema.names == ["y"]
