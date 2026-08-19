import json

import pandas as pd
import pytest

import framevitals
from framevitals.result import AnalysisResult, ColumnResult


def _sample_result() -> AnalysisResult:
    return AnalysisResult({
        "dataset_id": "fv_test",
        "filename": "customers.csv",
        "analysis_mode": "quick",
        "artifacts_enabled": False,
        "profile": {
            "shape": {"rows": 4, "columns": 2},
            "columns": ["age", "city"],
            "dtypes": {"age": "float64", "city": "object"},
            "missing_counts": {"age": 1, "city": 0},
            "missing_percent": {"age": 25.0, "city": 0.0},
            "duplicate_rows": 0,
            "duplicate_percent": 0.0,
            "memory_usage_mb": 0.01,
            "numeric_summary": {"age": {"mean": 30.0}},
            "categorical_summary": {
                "city": {"unique_values": 2, "top_values": {"Pune": 3}},
            },
            "correlations": {},
        },
        "column_roles": {
            "age": {
                "roles": ["numeric", "analysis_candidate"],
                "unique_count": 3,
                "unique_ratio": 0.75,
                "non_missing_count": 3,
                "is_numeric": True,
                "is_categorical": False,
            },
            "city": {
                "roles": ["categorical", "low_cardinality"],
                "unique_count": 2,
                "unique_ratio": 0.5,
                "non_missing_count": 4,
                "is_numeric": False,
                "is_categorical": True,
            },
        },
        "health": {
            "overall_score": 82.5,
            "label": "Good",
            "details": {"missing_percent": 12.5},
        },
        "ml_readiness": {"score": 72.0, "label": "Mostly Ready"},
        "signals": [
            {
                "name": "Data Completeness",
                "status": "Review",
                "severity": "Medium",
                "evidence": "12.5% of dataset cells are missing.",
                "recommendation": "Review missing values.",
            },
            {
                "name": "Temporal Data",
                "status": "Detected",
                "severity": "Informational",
                "evidence": "A date column exists.",
                "recommendation": "Use temporal analysis.",
            },
        ],
        "timings_ms": {"total": 125.0},
    })


def test_analysis_result_remains_dict_compatible():
    result = _sample_result()

    assert isinstance(result, dict)
    assert result["health"]["overall_score"] == 82.5
    assert result.get("filename") == "customers.csv"
    assert result["result_schema_version"] == "1"


def test_findings_are_normalized_from_existing_signals():
    result = _sample_result()

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["code"] == "signal.data_completeness"
    assert finding["severity"] == "medium"
    assert finding["method"] == "signal_engine"
    assert result.recommendations == ["Review missing values."]


def test_summary_and_column_helpers():
    result = _sample_result()

    summary = result.summary()
    assert summary["shape"] == {"rows": 4, "columns": 2}
    assert summary["finding_count"] == 1
    assert summary["health"]["overall_score"] == 82.5

    column = result.column("age")
    assert isinstance(column, ColumnResult)
    assert column.name == "age"
    assert column.missing_percent == 25.0
    assert column.numeric_summary["mean"] == 30.0

    with pytest.raises(KeyError, match="Column not found"):
        result.column("does_not_exist")


def test_json_export_is_full_and_round_trips(tmp_path):
    result = _sample_result()

    rendered = result.to_json()
    payload = json.loads(rendered)
    assert payload["profile"]["columns"] == ["age", "city"]
    assert payload["findings"][0]["code"] == "signal.data_completeness"

    destination = tmp_path / "nested" / "report.json"
    returned = result.to_json(destination)
    assert returned == destination
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["dataset_id"] == "fv_test"


def test_terminal_and_html_renderers(tmp_path):
    result = _sample_result()

    terminal = result.summary_text()
    assert "FrameVitals Analysis" in terminal
    assert "customers.csv" in terminal
    assert "Data Completeness" in terminal

    html = result.to_html()
    assert "<!doctype html>" in html.lower()
    assert "customers.csv" in html
    assert "Data Completeness" in html
    assert "Inspect raw JSON" in html

    destination = tmp_path / "report.html"
    returned = result.to_html(destination)
    assert returned == destination
    assert destination.exists()


def test_public_analyze_returns_analysis_result():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "city": ["Pune", "Mumbai", "Pune", "Nashik"],
    })

    result = framevitals.analyze(df, mode="quick")

    assert isinstance(result, framevitals.AnalysisResult)
    assert isinstance(result, dict)
    assert result["profile"]["shape"]["rows"] == 4
    assert result.result_schema_version == "1"
