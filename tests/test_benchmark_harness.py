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
