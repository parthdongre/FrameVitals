import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import framevitals
import framevitals.analysis_api as analysis_api
import framevitals.pipeline as materialized_pipeline


def test_streaming_analysis_does_not_reenter_materialized_core(tmp_path, monkeypatch):
    path = tmp_path / "bounded-only.parquet"
    frame = pd.DataFrame({
        "x": range(2_000),
        "y": [value * 2 for value in range(2_000)],
        "group": [f"g-{value % 5}" for value in range(2_000)],
    })
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=333)

    def fail(*args, **kwargs):
        raise AssertionError("streaming analysis re-entered materialized core work")

    # A streaming file must not fall back to the materialized public dispatcher.
    monkeypatch.setattr(analysis_api, "run_full_analysis", fail)
    # Nor may the bounded scheduler recompute core pandas profile/role/health/quality state.
    monkeypatch.setattr(materialized_pipeline, "build_profile", fail)
    monkeypatch.setattr(materialized_pipeline, "infer_column_roles", fail)
    monkeypatch.setattr(materialized_pipeline, "calculate_health_score", fail)
    monkeypatch.setattr(materialized_pipeline, "calculate_ml_readiness", fail)
    monkeypatch.setattr(materialized_pipeline, "run_quality_diagnostics", fail)

    result = framevitals.analyze(path, mode="quick", artifacts=False, workers=2)

    scheduler = result["execution"]["bounded_scheduler"]
    assert scheduler["enabled"] is True
    assert scheduler["core_reprofiled"] is False
    assert scheduler["source_budget_scope"] == "source_shape"
    assert scheduler["parallelism_budget_scope"] == "bounded_sample"
    assert result["profile"]["shape"] == {"rows": 2_000, "columns": 3}
    assert result["execution"]["streaming"]["full_materialization"] is False
    assert "profile" not in result["timings_ms"]
    assert result["timings_ms"]["streaming_profile"] > 0


def test_streaming_bounded_scheduler_uses_source_budget_for_ultra_wide_limits(tmp_path):
    # The focused unit check is shape-only: the source budget must retain the
    # ultra-wide relationship cap even though the retained sample is narrower.
    from framevitals.execution import derive_execution_budget
    from framevitals.streaming_bounded_pipeline import run_streaming_bounded_modules

    sample = pd.DataFrame({f"c{i}": range(100) for i in range(32)})
    source_budget = derive_execution_budget(100_000, 10_000, mode="standard")
    payload = run_streaming_bounded_modules(
        sample,
        dataset_id="test",
        original_filename="<test>",
        analysis_mode="standard",
        target_column=None,
        parallel_workers=4,
        source_budget=source_budget,
        column_roles={},
        skip_ai=True,
        disabled_modules={
            "deep_statistics",
            "anomaly_detection",
            "time_series",
            "text_profile",
            "modeling",
            "explainability",
            "cleaning",
            "charts",
            "quality_diagnostics",
        },
    )

    budget = payload["execution"]["budget"]
    assert budget["rows"] == 100_000
    assert budget["columns"] == 10_000
    assert budget["relationship_pair_budget"] == 10
    assert payload["execution"]["bounded_scheduler"]["sample_columns"] == 32
