import numpy as np
import pandas as pd
import pytest

from framevitals.analysis_state import AnalysisState, NumericColumnState
from framevitals.budgeted_analysis import (
    run_budgeted_anomalies,
    run_budgeted_deep_statistics,
    run_budgeted_time_series,
)
from framevitals.execution import derive_execution_budget, deterministic_sample_frame


def test_large_budget_limits_memory_heavy_parallelism():
    budget = derive_execution_budget(100_000, 105, mode="standard")

    assert budget.scale_class == "large"
    assert budget.large_dataset is True
    assert budget.max_memory_heavy_parallelism == 1
    assert budget.bootstrap_sample_rows < budget.rows
    assert budget.time_series_sample_rows < budget.rows


def test_extreme_shape_is_classified_without_allocating_dataset():
    budget = derive_execution_budget(100_000_000, 100_000, mode="standard")

    assert budget.scale_class == "extreme"
    assert budget.ultra_wide_dataset is True
    assert budget.cells == 10_000_000_000_000
    assert budget.max_memory_heavy_parallelism == 1


def test_deterministic_sample_is_bounded_and_covers_endpoints():
    frame = pd.DataFrame({"x": np.arange(10_000)})
    sampled, metadata = deterministic_sample_frame(frame, 1_000)

    assert len(sampled) == 1_000
    assert sampled.iloc[0]["x"] == 0
    assert sampled.iloc[-1]["x"] == 9_999
    assert metadata["sampled"] is True
    assert metadata["source_rows"] == 10_000
    assert metadata["sample_rows"] == 1_000


def test_numeric_state_merge_matches_full_frame_statistics():
    frame = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, np.inf, 7.0, 9.0],
        "y": [10, 11, 12, 13, 14, 15, 16],
    })

    full = AnalysisState.from_frame(frame)
    left = AnalysisState.from_frame(frame.iloc[:3])
    right = AnalysisState.from_frame(frame.iloc[3:])
    merged = left.merge(right)

    assert merged.rows == full.rows
    assert merged.schema == full.schema
    for name in ("x", "y"):
        actual = merged.numeric[name]
        expected = full.numeric[name]
        assert actual.count == expected.count
        assert actual.missing == expected.missing
        assert actual.infinite == expected.infinite
        assert actual.mean == pytest.approx(expected.mean)
        assert actual.variance == pytest.approx(expected.variance)
        assert actual.minimum == expected.minimum
        assert actual.maximum == expected.maximum


def test_numeric_state_rejects_incompatible_schema_merge():
    left = AnalysisState.from_frame(pd.DataFrame({"x": [1, 2]}))
    right = AnalysisState.from_frame(pd.DataFrame({"x": [1.0, 2.0]}))

    with pytest.raises(ValueError, match="different schemas"):
        left.merge(right)


def test_deep_statistics_adapter_never_passes_unbounded_frame(monkeypatch):
    seen = {}

    def fake_deep(frame, max_pairs=20):
        seen["rows"] = len(frame)
        seen["pairs"] = max_pairs
        return {"available": True}

    monkeypatch.setattr(
        "framevitals.budgeted_analysis.run_deep_statistics_v2",
        fake_deep,
    )
    frame = pd.DataFrame({"x": np.arange(20_000), "y": np.arange(20_000)})
    budget = derive_execution_budget(len(frame), len(frame.columns), mode="standard")

    result = run_budgeted_deep_statistics(frame, budget=budget)

    assert seen["rows"] <= budget.bootstrap_sample_rows
    assert seen["pairs"] <= budget.relationship_pair_budget
    assert result["execution"]["sampled"] is True
    assert result["execution"]["source_rows"] == 20_000


def test_anomaly_adapter_discloses_sample_coverage(monkeypatch):
    seen = {}

    def fake_anomalies(frame, **kwargs):
        seen["rows"] = len(frame)
        return {"available": True, "n_rows_scored": len(frame)}

    monkeypatch.setattr(
        "framevitals.budgeted_analysis.fast_anomaly_scan",
        fake_anomalies,
    )
    frame = pd.DataFrame({"x": np.arange(20_000)})
    budget = derive_execution_budget(len(frame), 1, mode="standard")

    result = run_budgeted_anomalies(frame, budget=budget)

    assert seen["rows"] == budget.anomaly_sample_rows
    assert result["execution"]["coverage"] == "sample"
    assert result["execution"]["sample_rows"] == budget.anomaly_sample_rows
    assert result["execution"]["anomaly_strategy"] == "fast_robust_random_projection"


def test_research_anomaly_adapter_keeps_heavy_confirmation(monkeypatch):
    seen = {"classical": 0, "neural": 0}

    def fake_classical(frame, **kwargs):
        seen["classical"] += 1
        return {"available": True}

    def fake_neural(frame, **kwargs):
        seen["neural"] += 1
        return {"available": True}

    monkeypatch.setattr(
        "framevitals.budgeted_analysis.detect_anomalies_ensemble",
        fake_classical,
    )
    monkeypatch.setattr(
        "framevitals.budgeted_analysis.neural_reconstruction_anomalies",
        fake_neural,
    )
    frame = pd.DataFrame({"x": np.arange(200), "y": np.arange(200) * 2})
    budget = derive_execution_budget(len(frame), 2, mode="research")

    result = run_budgeted_anomalies(frame, budget=budget)

    assert seen == {"classical": 1, "neural": 1}
    assert result["execution"]["anomaly_strategy"] == "classical_ensemble_plus_neural_reconstruction"
    assert result["execution"]["neural_reconstruction_enabled"] is True


def test_time_series_adapter_preserves_order(monkeypatch):
    seen = {}

    def fake_time_series(frame, target_column=None):
        seen["values"] = frame["x"].tolist()
        return {"available": True}

    monkeypatch.setattr(
        "framevitals.budgeted_analysis.detect_and_analyze_time_series",
        fake_time_series,
    )
    frame = pd.DataFrame({"x": np.arange(20_000)})
    budget = derive_execution_budget(len(frame), 1, mode="standard")

    result = run_budgeted_time_series(frame, budget=budget)

    assert seen["values"] == sorted(seen["values"])
    assert result["execution"]["temporal_order_preserved"] is True
    assert result["execution"]["sample_rows"] == budget.time_series_sample_rows


def test_numeric_column_state_empty_values_are_safe():
    state = NumericColumnState.from_series(pd.Series([np.nan, np.inf, -np.inf]))

    assert state.count == 0
    assert state.missing == 1
    assert state.infinite == 2
    assert state.mean == 0.0
    assert state.variance is None
