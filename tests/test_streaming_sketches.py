import numpy as np
import pytest

from framevitals.streaming_sketches import (
    NumpyLogQuantileSketch,
    PYTHON_NUMERIC_SKETCH_CELL_BUDGET,
    should_use_full_stream_numpy_sketch,
)


def test_numpy_log_quantile_sketch_is_mergeable_and_bounded():
    values = np.arange(1, 10_001, dtype=np.float64)
    left = NumpyLogQuantileSketch().update(values[:5_000])
    right = NumpyLogQuantileSketch().update(values[5_000:])
    merged = left.merge(right)

    assert merged.count == 10_000
    assert merged.quantile(0.5) == pytest.approx(5_000, rel=0.03)
    assert merged.quantile(0.95) == pytest.approx(9_500, rel=0.03)
    assert merged.bin_count < 1_000
    snapshot = merged.snapshot()
    assert snapshot["method"] == "numpy_log_quantile_sketch"
    assert snapshot["relative_accuracy"] == 0.01


def test_numpy_log_quantile_sketch_handles_signed_zero_and_nonfinite_values():
    sketch = NumpyLogQuantileSketch().update(
        np.array([-100.0, -10.0, -0.0, 0.0, 10.0, 100.0, np.nan, np.inf])
    )

    assert sketch.count == 6
    assert sketch.zero_count == 2
    assert sketch.quantile(0.0) < 0
    assert sketch.quantile(1.0) > 0


def test_numpy_stream_sketch_budget_uses_full_stream_for_normal_numeric_workloads():
    assert should_use_full_stream_numpy_sketch(100_000, 100) is True
    assert 100_000 * 100 <= PYTHON_NUMERIC_SKETCH_CELL_BUDGET


def test_numpy_stream_sketch_budget_avoids_ultra_wide_cpu_regression():
    assert should_use_full_stream_numpy_sketch(100_000, 6_750) is False
    assert should_use_full_stream_numpy_sketch(100_000, 10_000) is False
