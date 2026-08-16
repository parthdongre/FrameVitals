"""Deterministic catastrophic-regression guardrails for FrameVitals profiling.

These checks are deliberately broader than microbenchmarks. Hosted CI hardware is
noisy, so the initial budgets only fail large time/RSS regressions. Raw measurements
are written as JSON so the ceilings can be tightened later from observed history.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import framevitals as fv


ROWS = 150_000
NUMERIC_COLUMNS = 8
CATEGORICAL_COLUMNS = 4


def _peak_rss_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes. CI is Linux, but keep the script
    # useful locally as well.
    if os.uname().sysname.lower() == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def _frame(rows: int = ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    payload: dict[str, Any] = {}
    for index in range(NUMERIC_COLUMNS):
        values = rng.normal(loc=index, scale=1.0 + index / 10.0, size=rows)
        values[::997] = np.nan
        payload[f"value_{index}"] = values
    for index in range(CATEGORICAL_COLUMNS):
        payload[f"group_{index}"] = np.take(
            np.array([f"g{index}-{item}" for item in range(12)], dtype=object),
            np.arange(rows) % 12,
        )
    return pd.DataFrame(payload)


def _measure(name: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    before_rss = _peak_rss_mb()
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    after_rss = _peak_rss_mb()
    return {
        "name": name,
        "elapsed_seconds": round(float(elapsed), 6),
        "peak_rss_mb": round(float(after_rss), 3),
        "rss_growth_mb": round(max(0.0, float(after_rss - before_rss)), 3),
        "result": result,
    }


def _core_profile() -> dict[str, Any]:
    frame = _frame()
    profile = fv.profile(frame)
    assert profile["shape"] == {"rows": ROWS, "columns": NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}
    return {
        "rows": ROWS,
        "columns": NUMERIC_COLUMNS + CATEGORICAL_COLUMNS,
        "diagnostic": profile.diagnostic,
    }


def _parquet_profile() -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - workflow installs arrow
        raise RuntimeError(
            'Parquet guardrail requires: pip install "framevitals[arrow]"'
        ) from exc

    frame = _frame()
    with tempfile.TemporaryDirectory(prefix="framevitals-perf-") as directory:
        path = Path(directory) / "guardrail.parquet"
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            path,
            row_group_size=8_192,
        )
        del frame
        profile = fv.profile(path)

    assert profile["shape"] == {"rows": ROWS, "columns": NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}
    streaming = profile.get("streaming_metadata", {})
    assert streaming.get("enabled") is True
    assert streaming.get("full_materialization") is False
    return {
        "rows": ROWS,
        "columns": NUMERIC_COLUMNS + CATEGORICAL_COLUMNS,
        "diagnostic": profile.diagnostic,
        "streaming": True,
    }


SCENARIOS: dict[str, Callable[[], dict[str, Any]]] = {
    "core_profile_150k_x12": _core_profile,
    "parquet_profile_150k_x12": _parquet_profile,
}


def _load_budgets(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1":
        raise ValueError("Unsupported performance budget schema version.")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("Performance budget file is missing scenarios.")
    return scenarios


def run_guardrails(budget_path: Path) -> dict[str, Any]:
    budgets = _load_budgets(budget_path)
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for name, operation in SCENARIOS.items():
        if name not in budgets:
            raise ValueError(f"Missing performance budget for scenario: {name}")
        budget = budgets[name]
        measurement = _measure(name, operation)
        max_seconds = float(budget["max_seconds"])
        max_peak_rss_mb = float(budget["max_peak_rss_mb"])
        measurement["budget"] = {
            "max_seconds": max_seconds,
            "max_peak_rss_mb": max_peak_rss_mb,
        }
        measurement["passed"] = (
            measurement["elapsed_seconds"] <= max_seconds
            and measurement["peak_rss_mb"] <= max_peak_rss_mb
        )
        measurements.append(measurement)

        if measurement["elapsed_seconds"] > max_seconds:
            failures.append({
                "scenario": name,
                "metric": "elapsed_seconds",
                "observed": measurement["elapsed_seconds"],
                "budget": max_seconds,
            })
        if measurement["peak_rss_mb"] > max_peak_rss_mb:
            failures.append({
                "scenario": name,
                "metric": "peak_rss_mb",
                "observed": measurement["peak_rss_mb"],
                "budget": max_peak_rss_mb,
            })

    return {
        "schema_version": "1",
        "rows": ROWS,
        "scenario_count": len(measurements),
        "passed": not failures,
        "measurements": measurements,
        "failures": failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        type=Path,
        default=Path(__file__).with_name("performance_budgets.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_guardrails(args.budgets)
    rendered = json.dumps(result, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
