import numpy as np
import pandas as pd

from framevitals.budgeted_analysis import run_budgeted_deep_statistics
from framevitals.deep_triage import triage_deep_columns
from framevitals.execution import derive_execution_budget


def _wide_frame(rows: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    payload: dict[str, object] = {}
    for index in range(30):
        payload[f"n{index}"] = rng.normal(size=rows)
    payload["n27"] = rng.lognormal(mean=0.0, sigma=2.0, size=rows)
    payload["n28"] = np.where(np.arange(rows) % 4 == 0, np.nan, rng.normal(size=rows))
    payload["n29"] = np.ones(rows)
    for index in range(15):
        payload[f"c{index}"] = np.array([f"g-{value % (index + 2)}" for value in range(rows)], dtype=object)
    payload["c14"] = np.where(np.arange(rows) % 3 == 0, None, "rare")
    return pd.DataFrame(payload)


def test_deep_triage_bounds_columns_and_prioritizes_diagnostic_signals():
    frame = _wide_frame()
    result = triage_deep_columns(frame, mode="deep")

    assert len(result.selected_numeric) == 12
    assert len(result.selected_categorical) == 8
    assert {"n27", "n28", "n29"} <= set(result.selected_numeric)
    assert "c14" in result.selected_categorical
    metadata = result.to_dict()
    assert metadata["numeric_truncated"] is True
    assert metadata["categorical_truncated"] is True


def test_research_mode_keeps_a_larger_deep_diagnostic_surface():
    frame = _wide_frame()
    deep = triage_deep_columns(frame, mode="deep")
    research = triage_deep_columns(frame, mode="research")

    assert len(research.selected_numeric) > len(deep.selected_numeric)
    assert len(research.selected_categorical) > len(deep.selected_categorical)
    assert set(deep.selected_numeric) <= set(research.selected_numeric)


def test_budgeted_deep_statistics_only_passes_triaged_columns(monkeypatch):
    frame = _wide_frame(rows=4_000)
    seen: dict[str, object] = {}

    def fake_deep(diagnostic_view, max_pairs=20):
        seen["columns"] = list(diagnostic_view.columns)
        seen["rows"] = len(diagnostic_view)
        seen["pairs"] = max_pairs
        return {"available": True}

    monkeypatch.setattr(
        "framevitals.budgeted_analysis.run_deep_statistics_v2",
        fake_deep,
    )
    budget = derive_execution_budget(len(frame), len(frame.columns), mode="deep")
    result = run_budgeted_deep_statistics(frame, budget=budget)

    assert len(seen["columns"]) <= 20
    assert seen["rows"] <= budget.bootstrap_sample_rows
    assert result["column_triage"]["numeric_limit"] == 12
    assert result["column_triage"]["categorical_limit"] == 8
    assert result["execution"]["method"] == "adaptive_deep_statistics"
    assert result["execution"]["diagnostic_columns"] == len(seen["columns"])
