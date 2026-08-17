import numpy as np
import pandas as pd
import pytest

from framevitals.analysis_state import NumericColumnState
from framevitals.backends import numeric_state


def _reference_series() -> pd.Series:
    return pd.Series([1.0, 2.0, 2.0, 3.0, 9.0, 12.0, np.nan, np.inf])


def test_numpy_numeric_state_matches_pandas_bias_corrected_shape():
    series = _reference_series()
    finite = series[np.isfinite(series.to_numpy(dtype="float64", na_value=np.nan))]

    state = numeric_state(series, backend="numpy")

    assert state["count"] == len(finite)
    assert state["missing"] == 1
    assert state["infinite"] == 1
    assert state["skewness"] == pytest.approx(float(finite.skew()), abs=1e-12)
    assert state["kurtosis"] == pytest.approx(float(finite.kurtosis()), abs=1e-12)
    assert state["m3"] != 0.0
    assert state["m4"] > 0.0


def test_numeric_column_state_partition_merge_preserves_shape_statistics():
    values = pd.Series(
        np.concatenate([
            np.linspace(-8.0, 3.0, 2_500),
            np.linspace(4.0, 40.0, 1_500) ** 1.15,
        ])
    )
    full = NumericColumnState.from_series(values)
    partitions = [
        NumericColumnState.from_series(chunk)
        for chunk in np.array_split(values, 7)
    ]
    merged = NumericColumnState()
    for partition in partitions:
        merged = merged.merge(partition)

    assert merged.count == full.count
    assert merged.mean == pytest.approx(full.mean, rel=1e-12, abs=1e-12)
    assert merged.m2 == pytest.approx(full.m2, rel=1e-10, abs=1e-10)
    assert merged.m3 == pytest.approx(full.m3, rel=1e-10, abs=1e-10)
    assert merged.m4 == pytest.approx(full.m4, rel=1e-10, abs=1e-10)
    assert merged.skewness == pytest.approx(float(values.skew()), rel=1e-10, abs=1e-10)
    assert merged.kurtosis == pytest.approx(float(values.kurtosis()), rel=1e-10, abs=1e-10)


def test_constant_state_has_undefined_shape_statistics():
    state = NumericColumnState.from_series(pd.Series([5.0, 5.0, 5.0, 5.0, 5.0]))

    assert state.std == 0.0
    assert state.skewness is None
    assert state.kurtosis is None
