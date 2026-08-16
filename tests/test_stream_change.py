import numpy as np
import pandas as pd

from framevitals.budgeted_analysis import run_budgeted_time_series
from framevitals.execution import derive_execution_budget
from framevitals.stream_change import PageHinkleyMeanShift, scan_ordered_mean_shift


def test_page_hinkley_detects_sustained_batch_mean_shift():
    rng = np.random.default_rng(42)
    detector = PageHinkleyMeanShift(threshold=8.0, min_updates=8)

    for value in rng.normal(0.0, 0.08, size=24):
        detector.update(float(value))
    assert detector.detected is False

    for value in rng.normal(2.5, 0.08, size=12):
        detector.update(float(value))
        if detector.detected:
            break

    snapshot = detector.snapshot()
    assert snapshot["detected"] is True
    assert snapshot["direction"] == "up"
    assert snapshot["detected_at_batch"] is not None
    assert snapshot["sufficient_batches"] is True


def test_page_hinkley_stays_quiet_on_stationary_batch_means():
    rng = np.random.default_rng(7)
    detector = PageHinkleyMeanShift(threshold=10.0, min_updates=8)

    for value in rng.normal(10.0, 0.15, size=80):
        detector.update(float(value))

    assert detector.detected is False
    assert detector.snapshot()["updates"] == 80


def test_page_hinkley_ignores_nonfinite_observations():
    detector = PageHinkleyMeanShift()
    detector.update(None)
    detector.update(float("nan"))
    detector.update(float("inf"))

    assert detector.count == 0
    assert detector.snapshot()["sufficient_batches"] is False


def test_ordered_mean_shift_scan_detects_series_regime_change():
    rng = np.random.default_rng(123)
    series = pd.Series(np.concatenate([
        rng.normal(0.0, 0.1, size=600),
        rng.normal(3.0, 0.1, size=600),
    ]))

    result = scan_ordered_mean_shift(series, windows=24)

    assert result["available"] is True
    assert result["detected"] is True
    assert result["direction"] == "up"
    assert result["windows"] == 24


def test_budgeted_time_series_attaches_mean_shift_when_series_is_detected(monkeypatch):
    rows = 1_200
    rng = np.random.default_rng(5)
    values = np.concatenate([
        rng.normal(0.0, 0.1, size=600),
        rng.normal(2.5, 0.1, size=600),
    ])
    frame = pd.DataFrame({
        "event_time": pd.date_range("2025-01-01", periods=rows, freq="h"),
        "value": values,
    })

    def fake_time_series(work, target_column=None):
        return {
            "available": True,
            "detected_date_column": "event_time",
            "numeric_column": "value",
        }

    monkeypatch.setattr(
        "framevitals.budgeted_analysis.detect_and_analyze_time_series",
        fake_time_series,
    )
    budget = derive_execution_budget(rows, 2, mode="standard")
    result = run_budgeted_time_series(frame, budget=budget)

    assert result["mean_shift"]["available"] is True
    assert result["mean_shift"]["detected"] is True
    assert result["execution"]["mean_shift_detection_enabled"] is True
    assert result["execution"]["method"] == "bounded_time_series"
    assert result["execution"]["scope"] == "bounded_time_series"
    assert result["execution"]["adaptive_strategy"] == "ordered_page_hinkley_mean_shift"
