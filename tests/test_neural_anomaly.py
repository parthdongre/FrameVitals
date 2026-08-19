import numpy as np
import pandas as pd

from framevitals.budgeted_analysis import run_budgeted_anomalies
from framevitals.execution import derive_execution_budget
from framevitals.neural_anomaly import neural_reconstruction_anomalies


def _frame(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(rows, 5))
    x[-5:] += 8.0
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(5)])
    frame["constant"] = 1.0
    return frame


def test_neural_reconstruction_detector_is_bounded_and_finds_shifted_rows():
    frame = _frame()
    result = neural_reconstruction_anomalies(
        frame,
        max_rows=500,
        max_columns=5,
        max_iter=20,
        top_k=15,
    )

    assert result["available"] is True
    assert result["sample_rows"] <= 500
    assert result["columns_used"] <= 5
    assert result["method"] == "bounded_mlp_reconstruction"
    assert result["architecture"][0] == result["architecture"][-1]
    assert result["top_rows"]


def test_neural_reconstruction_is_research_only(monkeypatch):
    seen = {"calls": 0}

    def fake_classical(frame, **kwargs):
        return {"available": True, "n_rows_scored": len(frame)}

    def fake_neural(frame, **kwargs):
        seen["calls"] += 1
        return {"available": True, "method": "fake_neural"}

    monkeypatch.setattr(
        "framevitals.budgeted_analysis.detect_anomalies_ensemble",
        fake_classical,
    )
    monkeypatch.setattr(
        "framevitals.budgeted_analysis.neural_reconstruction_anomalies",
        fake_neural,
    )

    frame = _frame(200)
    deep = derive_execution_budget(len(frame), len(frame.columns), mode="deep")
    deep_result = run_budgeted_anomalies(frame, budget=deep)
    assert seen["calls"] == 0
    assert "neural_reconstruction" not in deep_result
    assert deep_result["execution"]["neural_reconstruction_enabled"] is False

    research = derive_execution_budget(len(frame), len(frame.columns), mode="research")
    research_result = run_budgeted_anomalies(frame, budget=research)
    assert seen["calls"] == 1
    assert research_result["neural_reconstruction"]["available"] is True
    assert research_result["execution"]["neural_reconstruction_enabled"] is True
