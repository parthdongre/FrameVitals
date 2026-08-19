import numpy as np
import pandas as pd

from framevitals.execution import (
    _deterministic_stratified_positions,
    deterministic_sample_frame,
)
from framevitals.streaming_profile import _sample_positions


def test_stratified_positions_are_reproducible_sorted_and_cover_full_range():
    first = _deterministic_stratified_positions(10_000, 1_000)
    second = _deterministic_stratified_positions(10_000, 1_000)

    assert np.array_equal(first, second)
    assert len(first) == 1_000
    assert first[0] == 0
    assert first[-1] == 9_999
    assert np.all(np.diff(first) > 0)


def test_stratified_jitter_breaks_periodic_phase_locking():
    rows = 5_000
    period = 51
    frame = pd.DataFrame({"periodic_spike": (np.arange(rows) % period == 0).astype(float)})

    sampled, metadata = deterministic_sample_frame(frame, 50)

    source_rate = float(frame["periodic_spike"].mean())
    sampled_rate = float(sampled["periodic_spike"].mean())
    assert abs(sampled_rate - source_rate) < 0.15
    assert metadata["strategy"] == "deterministic_stratified_jitter"
    assert metadata["sample_rows"] == 50
    assert metadata["seed"] > 0


def test_streaming_and_materialized_paths_share_sampling_positions():
    expected = _deterministic_stratified_positions(20_000, 1_337)
    actual = _sample_positions(20_000, 1_337)

    assert np.array_equal(actual, expected)
