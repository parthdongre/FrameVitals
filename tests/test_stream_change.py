import numpy as np

from framevitals.stream_change import PageHinkleyMeanShift


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
