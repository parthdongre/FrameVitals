import pandas as pd
import pytest

import framevitals as fv
from framevitals.sources import DatasetMetadata, DelimitedTextSource


def test_inspect_source_reports_pandas_capabilities():
    frame = pd.DataFrame({
        "value": [1, 2, 3],
        "label": ["a", "b", "c"],
    })

    info = fv.inspect_source(frame)

    assert info["name"] == "<dataframe>"
    assert info["kind"] == "memory"
    assert info["format"] == "pandas"
    assert info["rows"] == 3
    assert info["columns"] == 2
    assert info["size_bytes"] > 0
    assert info["materialized"] is True
    assert info["supports_projection"] is True
    assert info["supports_streaming"] is False


def test_inspect_source_reports_csv_fallback_without_arrow(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    pd.DataFrame({"value": [1, 2, 3]}).to_csv(path, index=False)
    monkeypatch.setattr(DelimitedTextSource, "_pyarrow_csv", lambda self: None)

    info = fv.inspect_source(path)

    assert info["name"] == "data.csv"
    assert info["kind"] == "file"
    assert info["format"] == "csv"
    assert info["rows"] is None
    assert info["columns"] is None
    assert info["size_bytes"] == path.stat().st_size
    assert info["materialized"] is False
    assert info["supports_projection"] is False
    assert info["supports_streaming"] is False


def test_inspect_source_accepts_custom_dataset_source():
    class CustomSource:
        def inspect(self):
            return DatasetMetadata(
                name="custom",
                kind="remote",
                format="custom",
                rows=42,
                columns=4,
                size_bytes=None,
                materialized=False,
                supports_projection=True,
                supports_streaming=False,
            )

        def load(self):
            return pd.DataFrame({"value": [1]})

    info = fv.inspect_source(CustomSource())

    assert info == {
        "name": "custom",
        "kind": "remote",
        "format": "custom",
        "rows": 42,
        "columns": 4,
        "size_bytes": None,
        "materialized": False,
        "supports_projection": True,
        "supports_streaming": False,
    }


def test_inspect_source_rejects_unsupported_input():
    with pytest.raises(TypeError, match="dataset path"):
        fv.inspect_source(object())
