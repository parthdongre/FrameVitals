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


def test_public_health_streams_parquet_without_calling_load(tmp_path, monkeypatch):
    path = tmp_path / "health.parquet"
    frame = _write_parquet(path, rows=12_000)

    def fail_load(self):
        raise AssertionError("health must not materialize the complete Parquet file")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.health(path)

    assert result["dataset_name"] == "health.parquet"
    assert 0 <= result["overall_score"] <= 100
    assert result["execution"]["method"] == "streaming_profile_with_bounded_row_sample"
    assert result["execution"]["full_materialization"] is False
    assert result["execution"]["source_rows"] == len(frame)
    assert result["execution"]["sample_rows"] == len(frame)
    assert result["execution"]["components"]["completeness"] == "full_stream_exact"
    assert result["execution"]["components"]["outlier_safety"] == "exact"
    expected_missing = int(frame["value"].isna().sum()) / (len(frame) * 4) * 100
    assert result["details"]["missing_percent"] == pytest.approx(
        round(expected_missing, 2)
    )


def test_public_ml_readiness_streams_parquet_without_calling_load(tmp_path, monkeypatch):
    path = tmp_path / "ml-readiness.parquet"
    frame = _write_parquet(path, rows=12_000)

    def fail_load(self):
        raise AssertionError("ML readiness must not materialize the complete Parquet file")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.ml_readiness(path)

    assert result["dataset_name"] == "ml-readiness.parquet"
    assert result["numeric_columns"] == ["value", "other"]
    assert result["categorical_columns"] == ["group"]
    expected_missing = int(frame["value"].isna().sum()) / (len(frame) * 4) * 100
    assert result["issues"]["missing_percent"] == pytest.approx(
        round(expected_missing, 2)
    )
    execution = result["execution"]
    assert execution["method"] == "streaming_profile"
    assert execution["full_materialization"] is False
    assert execution["source_rows"] == len(frame)
    assert execution["sample_rows"] == len(frame)
    assert execution["components"]["missingness"] == "full_stream_exact"
    assert execution["components"]["column_groups"] == "schema_exact"
    assert execution["components"]["duplicate_rate"] == "exact"
    assert 0 <= result["score"] <= 100


def test_large_public_ml_readiness_discloses_duplicate_estimate(tmp_path, monkeypatch):
    path = tmp_path / "large-ml-readiness.parquet"
    frame = _write_parquet(path, rows=60_000)

    def fail_load(self):
        raise AssertionError("large ML readiness must stay on the streaming path")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.ml_readiness(path)

    execution = result["execution"]
    assert execution["full_materialization"] is False
    assert execution["source_rows"] == len(frame)
    assert execution["sample_rows"] == 50_000
    assert execution["components"]["duplicate_rate"] == "bounded_row_sample_estimate"


def test_large_public_health_discloses_bounded_outlier_estimate(tmp_path, monkeypatch):
    path = tmp_path / "large-health.parquet"
    frame = _write_parquet(path, rows=60_000)

    def fail_load(self):
        raise AssertionError("large health must stay on the streaming path")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.health(path)

    execution = result["execution"]
    assert execution["full_materialization"] is False
    assert execution["source_rows"] == len(frame)
    assert execution["sample_rows"] == 50_000
    assert execution["components"]["outlier_safety"] == "bounded_row_sample_estimate"
    assert execution["components"]["uniqueness"] == "full_stream_sample_estimate"


def test_public_quality_streams_parquet_and_marks_sample_candidates(tmp_path, monkeypatch):
    path = tmp_path / "quality.parquet"
    frame = _write_parquet(path, rows=12_000)
    frame["other_copy"] = frame["other"]
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=777)

    def fail_load(self):
        raise AssertionError("quality must not materialize the complete Parquet file")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.quality(path, max_sample_rows=1_000)

    assert result["dataset_name"] == "quality.parquet"
    assert result["rows"] == len(frame)
    assert result["columns"] == len(frame.columns)
    execution = result["execution"]
    assert execution["full_materialization"] is False
    assert execution["sampled"] is True
    assert execution["sample_rows"] == 1_000
    assert set(execution["candidate_only_checks"]) == {
        "primary_key_candidates",
        "duplicate_columns",
    }

    duplicate_candidates = result["duplicate_columns"]
    assert duplicate_candidates
    candidate = next(
        item
        for item in duplicate_candidates
        if {item["canonical_column"], *item["duplicate_columns"]}
        >= {"other", "other_copy"}
    )
    assert candidate["candidate_only"] is True
    assert candidate["confirmed_with_full_equality"] is False
    assert candidate["confirmation_scope"] == "bounded_row_sample"


def test_plan_reads_only_bounded_parquet_sample_and_uses_true_shape(tmp_path, monkeypatch):
    path = tmp_path / "plan.parquet"
    frame = _write_parquet(path, rows=12_000)

    def fail_load(self):
        raise AssertionError("plan must not materialize the complete Parquet file")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.plan(path, mode="standard")

    assert result["shape"] == {"rows": len(frame), "columns": 4}
    assert result["signals"]["row_count"] == len(frame)
    assert result["signals"]["column_count"] == 4
    assert result["execution_budget"]["rows"] == len(frame)
    planning = result["planning_data"]
    assert planning["materialized_full_dataset"] is False
    assert planning["full_scan"] is False
    assert planning["sampled"] is True
    assert planning["sample_rows"] == 5_000
    assert result["source"]["supports_streaming"] is True


def test_relationships_stream_numeric_projection_without_full_materialization(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "relationships.parquet"
    frame = _write_parquet(path, rows=12_000)

    def fail_load(self):
        raise AssertionError("relationships must not materialize the complete Parquet file")

    monkeypatch.setattr(ParquetSource, "load", fail_load)
    result = framevitals.relationships(path, max_sample_rows=256)

    assert result["available"] is True
    assert result["sample"]["source_rows"] == len(frame)
    assert result["sample"]["sample_rows"] <= 256
    assert result["sample"]["sampled"] is True
    assert result["sample"]["full_materialization"] is False
    assert result["sample"]["strategy"] == "streaming_evenly_spaced_global_rows"
    assert result["source"]["supports_projection"] is True
    edge_pairs = {(edge["source"], edge["target"]) for edge in result["edges"]}
    assert ("value", "other") in edge_pairs or ("other", "value") in edge_pairs
