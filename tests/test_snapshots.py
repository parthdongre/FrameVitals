import json

from framevitals.result import AnalysisResult
from framevitals.snapshots import compare_snapshots, load_snapshot


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
