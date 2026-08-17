import numpy as np
import pandas as pd

from framevitals.fast_deep_statistics import (
    fast_mean_ci,
    fast_median_ci,
    run_fast_deep_statistics_v2,
)


def test_fast_mean_ci_matches_normal_sample_location():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(loc=3.0, scale=1.5, size=2_000))

    result = fast_mean_ci(series)

    assert result["available"] is True
    assert result["method"] == "student_t"
    assert result["n_resamples"] == 0
    assert result["low"] < series.mean() < result["high"]


def test_fast_median_ci_is_distribution_free_and_contains_median():
    rng = np.random.default_rng(7)
    series = pd.Series(rng.lognormal(mean=0.3, sigma=1.1, size=2_000))

    result = fast_median_ci(series)

    assert result["available"] is True
    assert result["method"] == "distribution_free_order_statistic"
    assert result["n_resamples"] == 0
    assert result["rank_low"] < result["rank_high"]
    assert result["low"] <= series.median() <= result["high"]


def test_fast_deep_statistics_preserves_v2_shape():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "x": rng.normal(size=120),
        "y": rng.gamma(shape=2.0, scale=1.0, size=120),
        "group": np.where(np.arange(120) % 2, "A", "B"),
    })

    result = run_fast_deep_statistics_v2(frame, max_pairs=5)

    assert result["version"] == "v2"
    assert result["numeric_columns"] == ["x", "y"]
    assert result["categorical_columns"] == ["group"]
    assert result["numeric_statistics"]["x"]["bootstrap_mean_ci"]["n_resamples"] == 0
    assert result["numeric_statistics"]["x"]["bootstrap_median_ci"]["n_resamples"] == 0
    assert result["inference"]["bootstrap_resamples"] == 0


def test_fast_ci_rejects_tiny_samples_like_existing_bootstrap_contract():
    series = pd.Series(np.arange(10, dtype=float))

    assert fast_mean_ci(series) == {"available": False, "reason": "n<20"}
    assert fast_median_ci(series) == {"available": False, "reason": "n<20"}
