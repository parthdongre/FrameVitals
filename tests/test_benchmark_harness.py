import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def test_scale_benchmark_harness_emits_machine_readable_result(tmp_path):
    output = tmp_path / "benchmark.json"
    script = Path("benchmarks/benchmark_profile_scale.py")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--rows",
            "200",
            "--numeric-columns",
            "4",
            "--categorical-columns",
            "1",
            "--scenarios",
            "numpy",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark_schema_version"] == 1
    assert payload["workload"]["rows"] == 200
    assert payload["workload"]["total_columns"] == 5
    assert len(payload["measurements"]) == 1

    measurement = payload["measurements"][0]
    assert measurement["scenario"] == "numpy"
    assert measurement["elapsed_seconds"] >= 0
    assert measurement["peak_rss_mb"] > 0
    assert measurement["result"]["shape"] == {"rows": 200, "columns": 5}
    assert json.loads(completed.stdout)["workload"] == payload["workload"]


def test_release_delta_benchmark_evidence_is_self_consistent():
    evidence = Path(
        "benchmarks/results/release_0.2.0_vs_0.1.0_10k_x64.json"
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))

    assert payload["benchmark_schema_version"] == 1
    assert payload["comparison"] == "FrameVitals 0.2.0 vs 0.1.0"
    assert payload["old_ref"] == (
        "v0.1.0@3da1432168fbfcb3dbe99fcfb6f6200f5e63214b"
    )
    assert payload["new_ref"] == (
        "develop/august@05b11e594995a3833ec08f5b4a8a145197bf4cab"
    )

    dataset = payload["dataset"]
    assert dataset["rows"] == 10_000
    assert dataset["columns"] == 64
    assert dataset["cells"] == 640_000
    assert dataset["format"] == "csv"

    methodology = payload["methodology"]
    assert methodology["warmups_per_version_mode"] == 1
    assert methodology["measured_repetitions_per_version_mode"] == 3
    assert methodology["measurement_order"] == "ABBAAB"
    assert methodology["scope"] == (
        "all 10,000 rows and all 64 columns in both releases"
    )

    measurements = payload["measurements"]
    assert len(measurements) == 12
    assert {(item["version"], item["mode"]) for item in measurements} == {
        ("0.1.0", "quick"),
        ("0.2.0", "quick"),
        ("0.1.0", "standard"),
        ("0.2.0", "standard"),
    }

    for mode in ("quick", "standard"):
        mode_result = payload["modes"][mode]
        old = mode_result["0.1.0"]
        new = mode_result["0.2.0"]

        for version_result in (old, new):
            assert version_result["profiled_columns"] == 64
            assert len(version_result["wall_seconds"]) == 3
            assert len(version_result["peak_rss_mb"]) == 3
            assert math.isclose(
                version_result["median_wall_seconds"],
                statistics.median(version_result["wall_seconds"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            assert math.isclose(
                version_result["median_peak_rss_mb"],
                statistics.median(version_result["peak_rss_mb"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )

        assert old["backend"]["selected"] == "legacy-python"
        assert old["backend"]["native_available"] is False
        assert new["backend"]["selected"] == "rust"
        assert new["backend"]["native_available"] is True

        expected_speedup = (
            old["median_wall_seconds"] / new["median_wall_seconds"]
        )
        expected_wall_reduction = (
            1.0
            - new["median_wall_seconds"] / old["median_wall_seconds"]
        ) * 100.0
        expected_rss_reduction = (
            1.0
            - new["median_peak_rss_mb"] / old["median_peak_rss_mb"]
        ) * 100.0

        assert math.isclose(
            mode_result["speedup_x"],
            expected_speedup,
            rel_tol=1e-12,
        )
        assert math.isclose(
            mode_result["wall_time_reduction_percent"],
            expected_wall_reduction,
            rel_tol=1e-12,
        )
        assert math.isclose(
            mode_result["peak_rss_reduction_percent"],
            expected_rss_reduction,
            rel_tol=1e-12,
        )

        assert new["median_wall_seconds"] < old["median_wall_seconds"]


def test_release_accuracy_evidence_matches_same_performance_dataset_and_contract():
    evidence = Path(
        "benchmarks/results/release_0.2.0_vs_0.1.0_accuracy_10k_x64.json"
    )
    payload = json.loads(evidence.read_text(encoding="utf-8"))

    assert payload["benchmark_schema_version"] == 1
    assert payload["comparison"] == (
        "FrameVitals 0.2.0 vs 0.1.0 same-dataset statistical accuracy"
    )
    assert payload["old_ref"] == (
        "v0.1.0@3da1432168fbfcb3dbe99fcfb6f6200f5e63214b"
    )

    dataset = payload["dataset"]
    assert dataset == {
        "bytes": 2_811_188,
        "cells": 640_000,
        "columns": 64,
        "format": "csv",
        "generator": "((row * (col + 3) + col * 17) % 2001 - 1000).astype(int16)",
        "rows": 10_000,
        "same_as_release_performance_run": 32010158292,
    }
    assert payload["tracked_columns"] == [
        "c000",
        "c001",
        "c007",
        "c031",
        "c063",
    ]
    assert payload["evidence_runs"] == {
        "accuracy_full_legacy_and_initial_native_run": 32014979365,
        "native_shape_corrected_run": 32015665811,
        "same_dataset_performance_run": 32010158292,
    }

    old = payload["versions"]["0.1.0"]
    new = payload["versions"]["0.2.0"]
    assert old["backend"] == {
        "native_available": False,
        "selected": "legacy-python",
    }
    assert new["backend"]["selected"] == "rust"
    assert new["backend"]["native_available"] is True

    old_summary = old["summary"]
    new_summary = new["summary"]
    unchanged_metrics = [
        "max_exact_fact_absolute_error",
        "max_mean_absolute_error",
        "max_std_absolute_error",
        "max_shape_absolute_error",
        "pearson_absolute_error",
    ]
    for metric in unchanged_metrics:
        assert math.isclose(
            new_summary[metric],
            old_summary[metric],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        assert math.isclose(
            payload["delta_new_minus_old"][metric],
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )

    assert old_summary["max_exact_fact_absolute_error"] == 0.0
    assert new_summary["max_exact_fact_absolute_error"] == 0.0
    assert old_summary["shape_values_unavailable"] == 0
    assert new_summary["shape_values_unavailable"] == 0

    quantiles = payload["quantile_error_context"]
    assert quantiles["tracked_quantile_values"] == 15
    assert quantiles["native_quantile_relative_accuracy_setting"] == 0.01
    assert math.isclose(
        new_summary["max_quantile_absolute_error"],
        quantiles["max_absolute_error_units"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        new_summary["mean_quantile_absolute_error"],
        quantiles["mean_absolute_error_units"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert quantiles["max_error_percent_of_column_range"] < 0.211
    assert quantiles["mean_error_percent_of_column_range"] < 0.080
