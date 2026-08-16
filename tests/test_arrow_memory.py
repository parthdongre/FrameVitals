import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

import framevitals
from framevitals.sources import ArrowTableSource, resolve_source


def _table(rows: int = 6_000):
    values = np.arange(rows, dtype=np.float64)
    values[::127] = np.nan
    return pa.table({
        "value": values,
        "other": np.arange(rows, dtype=np.float64) * 2.0,
        "group": [f"g-{index % 5}" for index in range(rows)],
    })


def test_arrow_table_source_exposes_exact_metadata_projection_and_batches():
    table = _table(2_500)
    source = resolve_source(table)

    assert isinstance(source, ArrowTableSource)
    metadata = source.inspect()
    assert metadata.name == "<arrow_table>"
    assert metadata.kind == "memory"
    assert metadata.format == "arrow"
    assert metadata.rows == table.num_rows
    assert metadata.columns == table.num_columns
    assert metadata.size_bytes == table.nbytes
    assert metadata.materialized is True
    assert metadata.supports_projection is True
    assert metadata.supports_streaming is True
    assert source.schema() == table.schema

    public_info = framevitals.inspect_source(table)
    assert public_info == metadata.to_dict()

    batches = list(source.iter_batches(batch_size=700, columns=["value", "group"]))
    assert sum(batch.num_rows for batch in batches) == table.num_rows
    assert max(batch.num_rows for batch in batches) <= 700
    assert batches[0].schema.names == ["value", "group"]


def test_public_profile_streams_arrow_table_without_pandas_materialization(monkeypatch):
    table = _table(12_000)

    def fail_load(self):
        raise AssertionError("Arrow profile must not materialize the complete table in pandas")

    monkeypatch.setattr(ArrowTableSource, "load", fail_load)
    result = framevitals.profile(table)

    assert result["dataset_name"] == "<arrow_table>"
    assert result["shape"] == {"rows": table.num_rows, "columns": table.num_columns}
    assert result["streaming_metadata"]["enabled"] is True
    assert result["streaming_metadata"]["full_materialization"] is False
    assert result["source_metadata"]["kind"] == "memory"
    assert result["source_metadata"]["format"] == "arrow"
    assert result["missing_counts"]["value"] == int(np.isnan(table["value"].to_numpy()).sum())


def test_public_analyze_dispatches_arrow_table_through_streaming_pipeline(monkeypatch):
    table = _table(6_000)

    def fail_load(self):
        raise AssertionError("Arrow analysis must not materialize the complete table in pandas")

    monkeypatch.setattr(ArrowTableSource, "load", fail_load)
    result = framevitals.analyze(table, mode="quick", artifacts=False, workers=1)

    assert result["filename"] == "<arrow_table>"
    assert result["profile"]["shape"] == {
        "rows": table.num_rows,
        "columns": table.num_columns,
    }
    assert result["execution"]["streaming"]["enabled"] is True
    assert result["execution"]["streaming"]["full_materialization"] is False
    assert result["execution"]["streaming"]["source_rows"] == table.num_rows


def test_arrow_record_batch_normalizes_to_streaming_source():
    batch = pa.record_batch({
        "value": [1.0, 2.0, 3.0],
        "label": ["a", "b", "a"],
    })
    source = resolve_source(batch)
    metadata = source.inspect()

    assert isinstance(source, ArrowTableSource)
    assert metadata.name == "<arrow_record_batch>"
    assert metadata.rows == 3
    assert metadata.columns == 2
    assert metadata.supports_streaming is True

    result = framevitals.profile(batch)
    assert result["dataset_name"] == "<arrow_record_batch>"
    assert result["shape"] == {"rows": 3, "columns": 2}


def test_arrow_capsule_table_normalizes_without_library_specific_adapter():
    table = _table(1_200)

    class CapsuleTable:
        def __arrow_c_stream__(self, requested_schema=None):
            return table.__arrow_c_stream__(requested_schema)

    source = resolve_source(CapsuleTable())
    metadata = source.inspect()

    assert isinstance(source, ArrowTableSource)
    assert metadata.name == "<arrow_capsule_table>"
    assert metadata.rows == table.num_rows
    assert metadata.columns == table.num_columns
    assert metadata.supports_streaming is True

    public_info = framevitals.inspect_source(CapsuleTable())
    assert public_info["format"] == "arrow"
    assert public_info["rows"] == table.num_rows
    assert public_info["supports_streaming"] is True

    result = framevitals.profile(CapsuleTable())
    assert result["shape"] == {"rows": table.num_rows, "columns": table.num_columns}
    assert result["source_metadata"]["format"] == "arrow"


def test_arrow_record_batch_reader_is_not_silently_materialized():
    table = _table(20)
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

    with pytest.raises(TypeError, match="cheap exact row count"):
        resolve_source(reader)
