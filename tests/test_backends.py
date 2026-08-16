import numpy as np
import pandas as pd
import pytest

from framevitals import backends
from framevitals.analysis_state import NumericColumnState


def test_numpy_numeric_state_matches_reference_semantics():
    payload = backends.numeric_state(
        np.array([1.0, 2.0, np.nan, np.inf, 4.0], dtype=np.float64),
        backend="numpy",
    )

    assert payload["backend"] == "numpy"
    assert payload["observations"] == 5
    assert payload["count"] == 3
    assert payload["missing"] == 1
    assert payload["infinite"] == 1
    assert payload["mean"] == pytest.approx(7.0 / 3.0)
    assert payload["variance"] == pytest.approx(7.0 / 3.0)
    assert payload["minimum"] == 1.0
    assert payload["maximum"] == 4.0


def test_auto_backend_falls_back_without_native_extension(monkeypatch):
    monkeypatch.setattr(backends, "native_available", lambda: False)
    assert backends.resolve_numeric_backend("auto") == "numpy"


def test_explicit_rust_backend_requires_native_extension(monkeypatch):
    monkeypatch.setattr(backends, "native_available", lambda: False)
    with pytest.raises(RuntimeError, match="Rust backend was requested"):
        backends.resolve_numeric_backend("rust")


def test_backend_environment_override(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_BACKEND", "numpy")
    assert backends.resolve_numeric_backend() == "numpy"


def test_numeric_profile_numpy_discloses_missing_sketch_layer():
    payload = backends.numeric_profile([1.0, 2.0, 3.0], backend="numpy")
    assert payload["backend"] == "numpy"
    assert payload["sketches_available"] is False


def test_analysis_state_reconstructs_m2_from_backend_variance(monkeypatch):
    monkeypatch.setattr(
        "framevitals.analysis_state.numeric_state",
        lambda series: {
            "count": 3,
            "missing": 1,
            "infinite": 0,
            "mean": 2.0,
            "variance": 1.0,
            "std": 1.0,
            "minimum": 1.0,
            "maximum": 3.0,
        },
    )
    state = NumericColumnState.from_series(pd.Series([1.0, 2.0, 3.0, None]))

    assert state.count == 3
    assert state.missing == 1
    assert state.mean == 2.0
    assert state.m2 == 2.0
    assert state.variance == 1.0
