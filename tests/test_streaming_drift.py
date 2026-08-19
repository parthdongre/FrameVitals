import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import framevitals
from framevitals.sources import ParquetSource


def _write_drift_parquet(path, *, rows: int, shift: float = 0.0) -> pd.DataFrame:
    frame = pd.DataFrame({
        "value": np.arange(rows, dtype=np.float64) + shift,
        "segment": [f"s-{index % 5}" for index in range(rows)],
    })
    frame.loc[::211, "value"] = np.nan
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        row_group_size=777,
    )
    return frame


def test_compare_streams_large_parquet_and_reports_true_source_shapes(
    tmp_path,
    monkeypatch,
):
    reference_path = tmp_path / "reference.parquet"
    current_path = tmp_path / "current.parquet"
    reference = _write_drift_parquet(reference_path, rows=60_000)
    current = _write_drift_parquet(current_path, rows=72_000, shift=250.0)

    def fail_load(self):
        raise AssertionError("drift comparison must not fully materialize Parquet inputs")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.compare(reference_path, current_path)

    assert result["available"] is True
    assert result["reference_name"] == "reference.parquet"
    assert result["current_name"] == "current.parquet"
    assert result["ref_shape"] == [len(reference), len(reference.columns)]
    assert result["cur_shape"] == [len(current), len(current.columns)]
    assert result["row_count_change_percent"] == pytest.approx(20.0)

    execution = result["execution"]
    assert execution["method"] == "bounded_source_compare"
    assert execution["full_materialization"] is False
    assert execution["sample_limit_rows_per_source"] == 50_000
    assert execution["reference"]["source_rows"] == len(reference)
    assert execution["reference"]["sample_rows"] == 50_000
    assert execution["reference"]["sampled"] is True
    assert execution["current"]["source_rows"] == len(current)
    assert execution["current"]["sample_rows"] == 50_000
    assert execution["current"]["sampled"] is True
    assert execution["components"]["source_shape"] == "exact"
    assert execution["components"]["value_distributions"] == "bounded_row_sample"
    assert execution["components"]["missingness"] == "bounded_row_sample"


def test_reference_only_gate_reuses_streaming_drift_path(tmp_path, monkeypatch):
    reference_path = tmp_path / "gate-reference.parquet"
    current_path = tmp_path / "gate-current.parquet"
    _write_drift_parquet(reference_path, rows=12_000)
    _write_drift_parquet(current_path, rows=12_000, shift=25.0)

    def fail_load(self):
        raise AssertionError("reference-only gate must not materialize Parquet inputs")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.gate(current_path, reference=reference_path)

    assert result["checks_run"] == ["drift"]
    assert result["execution"]["validation"] is None
    drift_execution = result["execution"]["drift"]
    assert drift_execution["full_materialization"] is False
    assert drift_execution["reference"]["source_rows"] == 12_000
    assert drift_execution["current"]["source_rows"] == 12_000
    assert drift_execution["reference"]["strategy"] == "full_stream_via_batches"
    assert drift_execution["current"]["strategy"] == "full_stream_via_batches"
