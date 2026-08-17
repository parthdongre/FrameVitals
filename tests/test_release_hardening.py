import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from framevitals import profiler
from framevitals.profiler import build_profile


def test_materialized_duplicate_estimate_uses_stratified_jitter(monkeypatch):
    frame = pd.DataFrame({"value": [1, 1, 2, 2, 3, 3, 4, 4]})
    seen = {}
    def positions(rows, target_rows):
        seen["args"] = (rows, target_rows)
        return np.array([0, 2, 4, 6], dtype=np.int64)
    monkeypatch.setattr(profiler, "MAX_EXACT_DUPLICATE_CELLS", 1)
    monkeypatch.setattr(profiler, "DUPLICATE_SAMPLE_ROWS", 4)
    monkeypatch.setattr(profiler, "_deterministic_stratified_positions", positions)
    _, metadata = profiler._duplicate_profile(frame)
    assert seen["args"] == (len(frame), 4)
    assert metadata["sampled"] is True
    assert metadata["strategy"] == "stratified_jitter_global_rows"


def test_pandas_profile_excludes_infinities_from_finite_moments_without_hiding_missingness(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_BACKEND", "numpy")
    frame = pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf, np.nan], "y": [2.0, 4.0, 8.0, 16.0, 32.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        profile = build_profile(frame)
    x = profile["numeric_summary"]["x"]
    assert x["count"] == 2.0
    assert x["mean"] == pytest.approx(1.5)
    assert x["min"] == pytest.approx(1.0)
    assert x["max"] == pytest.approx(2.0)
    assert profile["missing_counts"]["x"] == 1
    assert profile["numeric_summary_metadata"]["finite_only_moments"] is True
    assert profile["correlations"]["x"]["y"] == pytest.approx(1.0)


def test_pdf_report_builder_contains_only_framevitals_branding():
    import framevitals.pdf_report_builder as report_builder
    source = inspect.getsource(report_builder)
    assert "DataLens" not in source
    assert "DATALENS" not in source
    assert "FrameVitals Dataset Report" in source
