import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

import framevitals
from framevitals.sources import DelimitedTextSource, resolve_source


def _frame(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame({
        "value": np.arange(rows, dtype=np.float64),
        "other": np.arange(rows, dtype=np.float64) * 1.5,
        "group": [f"g-{index % 6}" for index in range(rows)],
    })
    frame.loc[::137, "value"] = np.nan
    return frame


def test_csv_source_exposes_exact_streaming_metadata_and_projection(tmp_path):
    path = tmp_path / "dataset.csv"
    frame = _frame(4_000)
    frame.to_csv(path, index=False)

    source = resolve_source(path)
    assert isinstance(source, DelimitedTextSource)

    metadata = source.inspect()
    assert metadata.format == "csv"
    assert metadata.rows == len(frame)
    assert metadata.columns == len(frame.columns)
    assert metadata.materialized is False
    assert metadata.supports_projection is True
    assert metadata.supports_streaming is True

    batches = list(source.iter_batches(batch_size=700, columns=["value", "group"]))
    assert sum(batch.num_rows for batch in batches) == len(frame)
    assert max(batch.num_rows for batch in batches) <= 700
    assert batches[0].schema.names == ["value", "group"]


def test_public_profile_streams_csv_without_calling_load(tmp_path, monkeypatch):
    path = tmp_path / "stream.csv"
    frame = _frame(12_000)
    frame.to_csv(path, index=False)

    def fail_load(self):
        raise AssertionError("streaming CSV profile must not materialize the complete file")

    monkeypatch.setattr(DelimitedTextSource, "load", fail_load)
    result = framevitals.profile(path)

    assert result["dataset_name"] == "stream.csv"
    assert result["shape"] == {"rows": len(frame), "columns": len(frame.columns)}
    assert result["streaming_metadata"]["enabled"] is True
    assert result["streaming_metadata"]["full_materialization"] is False
    assert result["source_metadata"]["format"] == "csv"
    assert result["source_metadata"]["supports_streaming"] is True
    assert result["missing_counts"]["value"] == int(frame["value"].isna().sum())


def test_public_analyze_dispatches_csv_through_streaming_pipeline(tmp_path, monkeypatch):
    path = tmp_path / "analyze.csv"
    frame = _frame(6_000)
    frame.to_csv(path, index=False)

    def fail_load(self):
        raise AssertionError("streaming CSV analysis must not materialize the complete file")

    monkeypatch.setattr(DelimitedTextSource, "load", fail_load)
    result = framevitals.analyze(path, mode="quick", artifacts=False, workers=1)

    assert result["filename"] == "analyze.csv"
    assert result["profile"]["shape"] == {"rows": len(frame), "columns": len(frame.columns)}
    assert result["execution"]["streaming"]["enabled"] is True
    assert result["execution"]["streaming"]["full_materialization"] is False
    assert result["execution"]["streaming"]["source_rows"] == len(frame)


def test_tsv_source_streams_with_tab_delimiter(tmp_path):
    path = tmp_path / "dataset.tsv"
    frame = _frame(1_500)
    frame.to_csv(path, index=False, sep="\t")

    source = resolve_source(path)
    assert isinstance(source, DelimitedTextSource)
    metadata = source.inspect()

    assert metadata.format == "tsv"
    assert metadata.rows == len(frame)
    assert metadata.columns == len(frame.columns)
    assert metadata.supports_streaming is True
    batch = next(source.iter_batches(batch_size=200, columns=["group"]))
    assert batch.schema.names == ["group"]
    assert batch.num_rows <= 200
