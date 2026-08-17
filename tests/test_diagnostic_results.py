import json

import numpy as np
import pandas as pd

import framevitals as fv


def _frame(rows: int = 120) -> pd.DataFrame:
    values = np.arange(rows, dtype=np.float64)
    return pd.DataFrame({
        "value": values,
        "other": values * 2.0 + 1.0,
        "group": [f"g-{index % 4}" for index in range(rows)],
        "target": [index % 2 for index in range(rows)],
    })


def test_profile_returns_dict_compatible_diagnostic_result_without_schema_mutation(tmp_path):
    result = fv.profile(_frame())

    assert isinstance(result, dict)
    assert isinstance(result, fv.DiagnosticResult)
    assert result.diagnostic == "profile"
    assert result.dataset_name == "<dataframe>"
    assert result.available is True
    assert "diagnostic" not in result

    detached = result.to_dict()
    assert type(detached) is dict
    assert detached == dict(result)
    detached["shape"]["rows"] = -1
    assert result["shape"]["rows"] == 120

    rendered = json.loads(result.to_json())
    assert rendered == dict(result)
    assert "diagnostic" not in rendered

    destination = tmp_path / "profile.json"
    returned = result.to_json(destination)
    assert returned == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == dict(result)


def test_focused_apis_label_their_diagnostic_result():
    frame = _frame()

    results = {
        "roles": fv.roles(frame),
        "health": fv.health(frame),
        "ml_readiness": fv.ml_readiness(frame),
        "quality": fv.quality(frame),
        "statistics": fv.statistics(frame, mode="quick", max_pairs=2),
        "anomalies": fv.anomalies(frame, mode="quick", max_columns=3, top_k=5),
        "relationships": fv.relationships(frame, max_sample_rows=64),
        "target_analysis": fv.target_analysis(frame, target="target"),
    }

    for diagnostic, result in results.items():
        assert isinstance(result, fv.DiagnosticResult)
        assert result.diagnostic == diagnostic
        assert result.dataset_name == "<dataframe>"
        assert "diagnostic" not in result


def test_diagnostic_result_execution_and_source_helpers_follow_public_provenance():
    result = fv.statistics(_frame(800), mode="quick", max_pairs=2)

    assert result.execution["execution_schema_version"] == "1"
    assert result.execution["method"] == "bounded_deep_statistics"
    assert result.source["format"] == "pandas"
    assert result.source["rows"] == 800

    summary = result.summary()
    assert summary["diagnostic"] == "statistics"
    assert summary["dataset_name"] == "<dataframe>"
    assert summary["method"] == "bounded_deep_statistics"
    assert summary["execution_schema_version"] == "1"
    assert summary["source_rows"] == 800
    assert "FrameVitals statistics" in result.summary_text()


def test_unavailable_diagnostic_result_exposes_available_false():
    frame = pd.DataFrame({"label": ["a", "b", "c", "d"]})
    result = fv.anomalies(frame, mode="quick")

    assert isinstance(result, fv.DiagnosticResult)
    assert result.available is False
    assert result["available"] is False
    assert result.execution["method"] == "bounded_anomaly_detection"
