import numpy as np
import pandas as pd
import pytest

from framevitals.anomaly_ensemble import detect_anomalies_ensemble


def _anomaly_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 80
    x = rng.normal(0, 1, rows)
    y = rng.normal(0, 1, rows)
    x[-1] = 12.0
    y[-1] = -10.0
    x[5] = np.inf
    y[7] = np.nan
    return pd.DataFrame({"x": x, "y": y})


def test_anomaly_ensemble_handles_infinity_and_reports_preparation():
    result = detect_anomalies_ensemble(_anomaly_frame(), top_k=5)

    assert result["available"] is True
    assert result["n_rows_scored"] == 80
    assert result["preparation"]["infinite_values_replaced"]["x"] == 1
    assert result["preparation"]["missing_values_imputed"]["x"] == 1
    assert result["preparation"]["missing_values_imputed"]["y"] == 1
    assert result["detectors_run"]
    assert isinstance(result["detectors_failed"], dict)
    assert isinstance(result["detectors_skipped"], dict)


def test_anomaly_ensemble_reports_detector_agreement_and_feature_context():
    result = detect_anomalies_ensemble(
        _anomaly_frame(),
        contamination=0.05,
        threshold=0.5,
        top_k=5,
    )

    assert result["available"] is True
    assert result["consensus"]["majority_detectors_required"] >= 1
    assert 0 <= result["consensus"]["flagged_fraction"] <= 1
    assert 0 <= result["flagged_fraction"] <= 1
    assert result["expected_anomaly_count"] == 4

    top = result["top_rows"][0]
    assert "agreement_count" in top
    assert "agreement_fraction" in top
    assert "flagged" in top
    assert top["top_feature_deviations"]
    assert {
        item["feature"] for item in top["top_feature_deviations"]
    }.issubset({"x", "y"})

    for detector in result["detectors_run"]:
        assert detector in result["detector_summaries"]
        assert detector in result["detector_vote_thresholds"]


def test_anomaly_ensemble_keeps_historical_score_fields():
    result = detect_anomalies_ensemble(_anomaly_frame())

    assert result["threshold"] == 0.6
    assert result["contamination"] == 0.05
    assert "flagged_count" in result
    assert "ensemble_summary" in result
    assert "top_rows" in result
    assert all("ensemble" in row for row in result["top_rows"])


def test_anomaly_ensemble_validates_public_controls():
    frame = _anomaly_frame()

    with pytest.raises(ValueError, match="threshold"):
        detect_anomalies_ensemble(frame, threshold=1.1)
    with pytest.raises(ValueError, match="top_k"):
        detect_anomalies_ensemble(frame, top_k=0)
    with pytest.raises(ValueError, match="max_columns"):
        detect_anomalies_ensemble(frame, max_columns=0)
