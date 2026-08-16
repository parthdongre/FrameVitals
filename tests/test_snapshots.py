import json

import pytest

from framevitals.result import AnalysisResult
from framevitals.snapshots import (
    SnapshotHistory,
    compare_snapshots,
    create_snapshot,
    load_snapshot,
)


def _result(*, columns=None, health=80.0, missing=None, findings=None):
    columns = columns or {"age": "int64", "city": "object"}
    missing = missing or {name: 0.0 for name in columns}
    return AnalysisResult({
        "dataset_id": "fv_runtime_id",
        "filename": "customers.csv",
        "analysis_mode": "quick",
        "profile": {
            "shape": {"rows": 10, "columns": len(columns)},
            "columns": list(columns),
            "dtypes": columns,
            "missing_percent": missing,
            "duplicate_percent": 0.0,
            "memory_usage_mb": 0.1,
        },
        "health": {
            "overall_score": health,
            "label": "Good",
            "components": {"completeness": health},
        },
        "ml_readiness": {
            "score": 75.0,
            "label": "Mostly Ready",
            "issues": {"missing_percent": max(missing.values(), default=0.0)},
        },
        "signals": [],
        "findings": findings or [],
        "config": {
            "mode": "quick",
            "target": None,
            "artifacts": False,
            "workers": 2,
        },
    })


def test_snapshot_is_compact_deterministic_state(tmp_path):
    first = _result().snapshot()
    second = _result().snapshot()

    assert first["snapshot_schema_version"] == "1"
    assert first["fingerprint"] == second["fingerprint"]
    assert "profile" not in first
    assert first["state"]["dataset"]["dtypes"]["age"] == "int64"

    path = tmp_path / "baseline.json"
    returned = _result().snapshot(path)
    assert path.exists()
    assert returned["fingerprint"] == load_snapshot(path)["fingerprint"]


def test_snapshot_diff_reports_schema_health_missingness_and_findings():
    reference = _result(
        health=85.0,
        findings=[{"code": "signal.duplicate_records"}],
    ).snapshot()
    current = _result(
        columns={"age": "float64", "city": "object", "segment": "object"},
        health=76.5,
        missing={"age": 10.0, "city": 0.0, "segment": 5.0},
        findings=[{"code": "signal.data_completeness"}],
    ).snapshot()

    diff = compare_snapshots(reference, current)

    assert diff["changed"] is True
    assert diff["schema"]["added_columns"] == ["segment"]
    assert diff["schema"]["type_changes"]["age"] == {
        "reference": "int64",
        "current": "float64",
    }
    assert diff["missingness_changes"]["age"]["delta"] == 10.0
    assert diff["health_delta"] == -8.5
    assert diff["findings"]["new"] == ["signal.data_completeness"]
    assert diff["findings"]["resolved"] == ["signal.duplicate_records"]


def test_snapshot_json_roundtrip(tmp_path):
    path = tmp_path / "snapshot.json"
    snapshot = _result().snapshot(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["fingerprint"] == snapshot["fingerprint"]
    loaded = load_snapshot(path)
    assert loaded.diff(snapshot)["changed"] is False


def test_snapshot_history_persists_orders_and_compares_latest(tmp_path):
    history = SnapshotHistory(tmp_path / "history")
    baseline = create_snapshot(_result(health=90.0))
    baseline["created_at"] = "2026-08-14T08:00:00+00:00"
    current = create_snapshot(
        _result(
            health=72.5,
            findings=[{"code": "quality.high_missingness.age"}],
        )
    )
    current["created_at"] = "2026-08-15T08:00:00+00:00"

    first_path = history.add(baseline, label="baseline release")
    second_path = history.add(current, label="production")

    assert len(history) == 2
    assert "baseline-release" in first_path.name
    assert "production" in second_path.name
    assert history.previous()["fingerprint"] == baseline["fingerprint"]
    assert history.latest()["fingerprint"] == current["fingerprint"]

    diff = history.compare_latest()
    assert diff["changed"] is True
    assert diff["health_delta"] == -17.5
    assert diff["findings"]["new"] == ["quality.high_missingness.age"]

    timeline = history.timeline()
    assert [row["health_score"] for row in timeline] == [90.0, 72.5]
    assert [row["finding_count"] for row in timeline] == [0, 1]
    assert all(row["filename"] == "customers.csv" for row in timeline)


def test_snapshot_history_can_add_analysis_result_directly(tmp_path):
    history = SnapshotHistory(tmp_path / "history")
    path = history.add(_result(health=88.0), label="nightly")

    assert path.exists()
    assert len(history) == 1
    assert history.latest()["state"]["health"]["overall_score"] == 88.0


def test_snapshot_history_requires_two_entries_for_latest_diff(tmp_path):
    history = SnapshotHistory(tmp_path / "history")
    history.add(_result())

    with pytest.raises(ValueError, match="at least two"):
        history.compare_latest()
