import json
from pathlib import Path
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
