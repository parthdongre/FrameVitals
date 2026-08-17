from framevitals.streaming_exact_reuse import reuse_streaming_exact_statistics


def test_reuses_full_stream_moments_without_overwriting_sample_diagnostics():
    payload = {
        "profile": {
            "numeric_summary": {
                "x": {
                    "count": 100,
                    "mean": 12.5,
                    "std": 3.25,
                    "min": -2.0,
                    "25%": 10.0,
                    "50%": 12.0,
                    "75%": 15.0,
                    "max": 30.0,
                    "skewness": 1.25,
                    "kurtosis": 2.0,
                }
            },
            "numeric_summary_metadata": {
                "backend": "rust",
                "method": "native_streaming_accumulator",
                "higher_moments": "full_stream_exact",
            },
            "streaming_metadata": {
                "enabled": True,
                "numeric_backend": "rust",
            },
        },
        "deep_statistics_v2": {
            "numeric_statistics": {
                "x": {
                    "count": 20,
                    "mean": 99.0,
                    "std": 77.0,
                    "min": 1.0,
                    "max": 200.0,
                    "median": 13.0,
                    "skewness": 0.4,
                    "skewness_label": "Approximately Symmetric",
                    "kurtosis": -2.0,
                    "kurtosis_label": "Light-tailed",
                    "distribution_fit": {"available": True, "best_fit": {"name": "norm"}},
                }
            },
            "execution": {"scope": "bounded_deep_statistics"},
        },
    }

    result = reuse_streaming_exact_statistics(payload)
    stats = result["deep_statistics_v2"]["numeric_statistics"]["x"]

    assert stats["count"] == 100
    assert stats["mean"] == 12.5
    assert stats["std"] == 3.25
    assert stats["min"] == -2.0
    assert stats["max"] == 30.0
    assert stats["median"] == 13.0
    assert stats["skewness"] == 1.25
    assert stats["skewness_label"] == "Highly Skewed"
    assert stats["kurtosis"] == 2.0
    assert stats["kurtosis_label"] == "Heavy-tailed"
    assert stats["distribution_fit"]["best_fit"]["name"] == "norm"
    assert stats["summary_provenance"]["backend"] == "rust"
    assert set(stats["summary_provenance"]["reused_exact_fields"]) == {
        "count",
        "mean",
        "std",
        "min",
        "max",
        "skewness",
        "kurtosis",
    }
    assert "skewness" not in stats["summary_provenance"]["sample_derived_fields"]
    assert "kurtosis" not in stats["summary_provenance"]["sample_derived_fields"]
    reuse = result["deep_statistics_v2"]["execution"]["exact_once_reuse"]
    assert reuse["enabled"] is True
    assert reuse["columns_reused"] == 1
    assert set(reuse["fields"]) >= {"skewness", "kurtosis"}


def test_noop_when_deep_statistics_are_not_applicable():
    payload = {
        "profile": {
            "numeric_summary": {"x": {"count": 10, "mean": 2.0}},
            "streaming_metadata": {"enabled": True, "numeric_backend": "rust"},
        },
        "deep_statistics_v2": None,
    }

    assert reuse_streaming_exact_statistics(payload) is payload
