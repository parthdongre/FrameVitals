import pandas as pd
import pytest

import framevitals.profiler as profiler


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "value": [1.0, 2.0, None, 4.0],
        "other": [10.0, 20.0, 30.0, 40.0],
        "label": ["a", "b", "a", "b"],
    })


def test_small_profile_keeps_exact_pandas_summary_when_native_is_available(monkeypatch):
    monkeypatch.delenv("FRAMEVITALS_BACKEND", raising=False)
    monkeypatch.setattr(profiler, "resolve_numeric_backend", lambda: "rust")

    result = profiler.build_profile(_frame())

    metadata = result["numeric_summary_metadata"]
    assert metadata["backend"] == "pandas"
    assert metadata["approximate_quantiles"] is False
    assert metadata["native_eligible"] is True
    assert metadata["native_threshold_reached"] is False
    assert result["numeric_summary"]["value"]["50%"] == pytest.approx(2.0)


def test_forced_native_profile_reuses_missing_counts_and_preserves_summary_shape(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_BACKEND", "rust")
    monkeypatch.setattr(profiler, "resolve_numeric_backend", lambda: "rust")

    calls = []

    def fake_numeric_profile(series, *, backend, stream_id):
        calls.append((series.name, backend, stream_id))
        if series.name == "value":
            return {
                "count": 3,
                "missing": 1,
                "infinite": 0,
                "mean": 7.0 / 3.0,
                "std": 1.527525,
                "minimum": 1.0,
                "maximum": 4.0,
                "quantiles": {
                    "p25": 1.5,
                    "p50": 2.0,
                    "p75": 3.0,
                    "relative_accuracy": 0.01,
                },
            }
        return {
            "count": 4,
            "missing": 0,
            "infinite": 0,
            "mean": 25.0,
            "std": 12.909944,
            "minimum": 10.0,
            "maximum": 40.0,
            "quantiles": {
                "p25": 17.5,
                "p50": 25.0,
                "p75": 32.5,
                "relative_accuracy": 0.01,
            },
        }

    monkeypatch.setattr(profiler, "numeric_profile", fake_numeric_profile)
    result = profiler.build_profile(_frame())

    assert calls == [("value", "rust", 0), ("other", "rust", 1)]
    assert result["missing_counts"]["value"] == 1
    assert result["missing_counts"]["other"] == 0
    assert set(result["numeric_summary"]["value"]) == {
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    }
    metadata = result["numeric_summary_metadata"]
    assert metadata["backend"] == "rust"
    assert metadata["approximate_quantiles"] is True
    assert metadata["quantile_relative_accuracy"] == 0.01
    assert metadata["raw_observations_retained"] is False


def test_auto_native_failure_falls_back_and_discloses_reason(monkeypatch):
    monkeypatch.delenv("FRAMEVITALS_BACKEND", raising=False)
    monkeypatch.setattr(profiler, "resolve_numeric_backend", lambda: "rust")
    monkeypatch.setattr(profiler, "NATIVE_NUMERIC_PROFILE_MIN_ROWS", 1)
    monkeypatch.setattr(
        profiler,
        "numeric_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(BufferError("not contiguous")),
    )

    result = profiler.build_profile(_frame())
    metadata = result["numeric_summary_metadata"]

    assert metadata["backend"] == "pandas"
    assert metadata["fallback_from"] == "rust"
    assert "BufferError" in metadata["fallback_reason"]


def test_forced_native_failure_is_not_silently_hidden(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_BACKEND", "rust")
    monkeypatch.setattr(profiler, "resolve_numeric_backend", lambda: "rust")
    monkeypatch.setattr(
        profiler,
        "numeric_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(BufferError("bad buffer")),
    )

    with pytest.raises(BufferError, match="bad buffer"):
        profiler.build_profile(_frame())
