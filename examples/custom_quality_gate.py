"""Example: combine FrameVitals contracts, drift, and domain-specific checks."""

from __future__ import annotations

import pandas as pd

import framevitals as fv


@fv.check(
    "positive revenue",
    severity="error",
    description="Accounting exports must never contain negative recognized revenue.",
)
def positive_revenue(df: pd.DataFrame):
    invalid = int((df["revenue"] < 0).sum())
    return {
        "passed": invalid == 0,
        "message": (
            "Revenue values are non-negative."
            if invalid == 0
            else f"Found {invalid} negative revenue record(s)."
        ),
        "details": {"negative_rows": invalid},
    }


@fv.check("preferred latency budget", severity="warning")
def latency_budget(df: pd.DataFrame):
    p95 = float(df["latency_ms"].quantile(0.95))
    return {
        "passed": p95 <= 250.0,
        "message": f"p95 latency is {p95:.1f} ms; preferred maximum is 250 ms.",
        "details": {"p95_latency_ms": p95},
    }


def main() -> None:
    reference = pd.read_parquet("data/training.parquet")
    current = pd.read_parquet("data/production.parquet")

    contract = fv.infer_contract(reference)
    result = fv.gate(
        current,
        reference=reference,
        contract=contract,
        custom_checks=[positive_revenue, latency_budget],
        drift_warn_on="moderate",
        drift_fail_on="severe",
    )

    print(result.summary_text())
    result.to_json("framevitals-gate.json")

    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
