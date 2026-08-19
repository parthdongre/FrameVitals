import numpy as np
import pandas as pd

from framevitals.fast_anomaly import fast_anomaly_scan


def test_fast_anomaly_scan_detects_injected_multivariate_shift():
    rng = np.random.default_rng(42)
    rows = 2_000
    matrix = rng.normal(size=(rows, 8))
    matrix[-12:] += 9.0
    frame = pd.DataFrame(matrix, columns=[f"x{i}" for i in range(8)])

    result = fast_anomaly_scan(frame, contamination=0.02, top_k=25)

    assert result["available"] is True
    assert result["method"] == "fast_robust_random_projection"
    assert result["detectors_run"] == [
        "robust_feature_deviation",
        "random_projection_tail",
        "random_projection_density",
    ]
    assert result["n_rows_scored"] == rows
    top_indices = {int(item["row_index"]) for item in result["top_rows"][:20]}
    assert len(top_indices & set(range(rows - 12, rows))) >= 8


def test_fast_anomaly_scan_bounds_numeric_dimensions():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(500, 80)))

    result = fast_anomaly_scan(frame, max_columns=16, projections=8)

    assert result["available"] is True
    assert len(result["used_columns"]) == 16
    assert result["preparation"]["truncated_columns"] is True
    assert result["projection_count"] == 8
