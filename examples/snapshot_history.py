"""Example: persist compact FrameVitals monitoring state across recurring runs."""

from __future__ import annotations

import framevitals as fv


def main() -> None:
    report = fv.analyze("data/production.parquet", mode="quick", artifacts=False)

    history = fv.SnapshotHistory(".framevitals/history")
    path = history.add(report, label="nightly")
    print(f"Stored snapshot: {path}")

    latest = history.latest()
    if latest is not None:
        print(f"Latest fingerprint: {latest['fingerprint']}")

    if len(history) >= 2:
        change = history.compare_latest()
        print(f"Health delta: {change['health_delta']}")
        print(f"New findings: {change['findings']['new']}")

    for point in history.timeline()[-5:]:
        print(
            point["created_at"],
            point["health_score"],
            point["ml_readiness_score"],
            point["finding_count"],
        )


if __name__ == "__main__":
    main()
