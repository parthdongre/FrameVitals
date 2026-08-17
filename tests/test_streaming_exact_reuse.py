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
                }
            },
            "numeric_summary_metadata": {
                "backend": "rust",
                "method": "native_streaming_accumulator",
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
    assert stats["skewness"] == 0.4
    assert stats["distribution_fit"]["best_fit"]["name"] == "norm"
    assert stats["summary_provenance"]["backend"] == "rust"
    assert set(stats["summary_provenance"]["reused_exact_fields"]) == {
        "count",
        "mean",
        "std",
        "min",
        "max",
    }
    reuse = result["deep_statistics_v2"]["execution"]["exact_once_reuse"]
    assert reuse["enabled"] is True
    assert reuse["columns_reused"] == 1


def test_noop_when_deep_statistics_are_not_applicable():
    payload = {
        "profile": {
            "numeric_summary": {"x": {"count": 10, "mean": 2.0}},
            "streaming_metadata": {"enabled": True, "numeric_backend": "rust"},
        },
        "deep_statistics_v2": None,
    }

    assert reuse_streaming_exact_statistics(payload) is payload
