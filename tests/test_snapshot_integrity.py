import json

import pytest

from framevitals.snapshots import SnapshotHistory, create_snapshot, load_snapshot


def _result():
    return {
        "dataset_id": "fv_test",
        "filename": "data.csv",
        "analysis_mode": "quick",
        "profile": {
            "shape": {"rows": 1, "columns": 1},
            "dtypes": {"x": "int64"},
            "missing_percent": {"x": 0.0},
        },
        "health": {"overall_score": 100.0},
        "ml_readiness": {"score": 100.0},
        "findings": [],
        "config": {},
    }


def test_load_snapshot_rejects_state_modified_after_fingerprinting(tmp_path):
    snapshot = create_snapshot(_result())
    snapshot["state"]["health"]["overall_score"] = 1.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_snapshot(path)


def test_history_rejects_tampered_existing_snapshot(tmp_path):
    snapshot = create_snapshot(_result())
    snapshot["state"]["analysis_mode"] = "research"

    with pytest.raises(ValueError, match="fingerprint"):
        SnapshotHistory(tmp_path).add(snapshot)
