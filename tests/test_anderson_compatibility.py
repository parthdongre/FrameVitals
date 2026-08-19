import inspect
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest

from framevitals import deep_statistics_v2
from framevitals.deep_statistics_v2 import _anderson_normality, _normality


def test_anderson_supported_api_is_warning_free():
    rng = np.random.default_rng(42)
    sample = pd.Series(rng.normal(size=256))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = _normality(sample)

    anderson = result["anderson"]
    assert anderson["statistic"] is not None

    if "method" in inspect.signature(deep_statistics_v2.stats.anderson).parameters:
        assert anderson["method"] == "interpolate"
        assert anderson["critical_5pct"] is None
        assert 0.0 <= anderson["p_value"] <= 1.0
    else:
        assert anderson["method"] == "legacy_critical_values"
        assert anderson["p_value"] is None
        assert anderson["critical_5pct"] is not None


def test_anderson_legacy_api_falls_back_to_critical_values(monkeypatch):
    calls = []

    def fake_anderson(values, dist="norm", **kwargs):
        calls.append(kwargs)
        if "method" in kwargs:
            raise TypeError("anderson() got an unexpected keyword argument 'method'")
        return SimpleNamespace(
            statistic=0.42,
            critical_values=np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
            significance_level=np.array([15.0, 10.0, 5.0, 2.5, 1.0]),
        )

    monkeypatch.setattr(deep_statistics_v2.stats, "anderson", fake_anderson)
    result = _anderson_normality(pd.Series(np.arange(20, dtype=float)))

    assert calls == [{"method": "interpolate"}, {}]
    assert result == {
        "statistic": 0.42,
        "p_value": None,
        "critical_5pct": 0.7,
        "method": "legacy_critical_values",
    }


def test_normality_verdict_still_prefers_shapiro(monkeypatch):
    sample = pd.Series(np.linspace(-2.0, 2.0, 100))

    monkeypatch.setattr(
        deep_statistics_v2.stats,
        "shapiro",
        lambda values: (0.99, 0.8),
    )
    monkeypatch.setattr(
        deep_statistics_v2.stats,
        "normaltest",
        lambda values: (20.0, 0.001),
    )
    monkeypatch.setattr(
        deep_statistics_v2,
        "_anderson_normality",
        lambda values: {
            "statistic": 2.0,
            "p_value": 0.001,
            "critical_5pct": None,
            "method": "interpolate",
        },
    )

    result = _normality(sample)

    assert result["shapiro"]["p_value"] == pytest.approx(0.8)
    assert result["dagostino"]["p_value"] == pytest.approx(0.001)
    assert result["anderson"]["p_value"] == pytest.approx(0.001)
    assert result["is_probably_normal"] is True
