import pandas as pd

from framevitals.cli import main
from framevitals.drift_analysis import compare_datasets, severity_at_least


def test_identical_datasets_produce_pass_gate():
    frame = pd.DataFrame({
        "value": list(range(30)),
        "group": ["A", "B", "C"] * 10,
        "event_time": pd.date_range("2026-01-01", periods=30, freq="D"),
    })

    result = compare_datasets(frame, frame.copy())

    assert result["available"] is True
    assert result["gate"]["status"] == "pass"
    assert result["summary"]["overall_verdict"] == "stable"
    assert result["schema"]["severity"] == "stable"


def test_added_schema_column_warns_without_failing():
    reference = pd.DataFrame({"value": list(range(30))})
    current = reference.assign(new_feature=1)

    result = compare_datasets(reference, current)

    assert result["schema"]["added_columns"] == ["new_feature"]
    assert result["schema"]["removed_columns"] == []
    assert result["schema"]["severity"] == "moderate"
    assert result["gate"]["status"] == "warn"


def test_removed_column_or_type_change_is_fail_level():
    reference = pd.DataFrame({
        "value": list(range(30)),
        "segment": ["A", "B"] * 15,
    })
    current = pd.DataFrame({
        "value": [str(value) for value in range(30)],
    })

    result = compare_datasets(reference, current)

    assert result["schema"]["removed_columns"] == ["segment"]
    assert result["schema"]["dtype_changes"][0]["column"] == "value"
    assert result["summary"]["overall_verdict"] == "severe"
    assert result["gate"]["status"] == "fail"


def test_missingness_change_contributes_to_column_severity():
    reference = pd.DataFrame({"value": list(range(50))})
    current_values = list(range(20)) + [None] * 30
    current = pd.DataFrame({"value": current_values})

    result = compare_datasets(reference, current)
    column = result["columns"][0]

    assert column["cur_missing_percent"] == 60.0
    assert column["missingness_delta_percentage_points"] == 60.0
    assert column["missingness_severity"] == "severe"
    assert column["drift_severity"] == "severe"


def test_categorical_drift_reports_jensen_shannon_distance():
    reference = pd.DataFrame({
        "city": ["Pune"] * 30 + ["Mumbai"] * 20,
    })
    current = pd.DataFrame({
        "city": ["Pune"] * 5 + ["Mumbai"] * 5 + ["Delhi"] * 40,
    })

    result = compare_datasets(reference, current)
    column = result["columns"][0]

    assert column["jensen_shannon_distance"] is not None
    assert column["jensen_shannon_distance"] > 0
    assert "Delhi" in column["new_categories"]
    assert column["drift_severity"] in {"moderate", "severe"}


def test_datetime_drift_is_analyzed_as_datetime():
    reference = pd.DataFrame({
        "event_time": pd.date_range("2026-01-01", periods=30, freq="D"),
    })
    current = pd.DataFrame({
        "event_time": pd.date_range("2026-06-01", periods=30, freq="D"),
    })

    result = compare_datasets(reference, current)
    column = result["columns"][0]

    assert column["type"] == "datetime"
    assert column["available"] is True
    assert column["wasserstein_normalized"] is not None
    assert column["ref_min"].startswith("2026-01-01")
    assert column["cur_min"].startswith("2026-06-01")


def test_requested_column_diagnostics_and_truncation_are_explicit():
    reference = pd.DataFrame({
        "a": list(range(30)),
        "b": list(range(30)),
        "c": list(range(30)),
    })
    current = reference.copy()

    result = compare_datasets(
        reference,
        current,
        columns=["a", "b", "c", "missing"],
        max_columns=2,
    )

    assert result["selection"]["truncated"] is True
    assert result["selection"]["total_selected_columns"] == 3
    assert result["selection"]["requested_missing_in_reference"] == ["missing"]
    assert result["selection"]["requested_missing_in_current"] == ["missing"]
    assert result["shared_columns"] == ["a", "b"]


def test_severity_threshold_helper_is_ordered():
    assert severity_at_least("severe", "moderate") is True
    assert severity_at_least("moderate", "moderate") is True
    assert severity_at_least("minor", "moderate") is False
    assert severity_at_least("stable", "minor") is False


def test_compare_cli_fail_on_is_opt_in(tmp_path, monkeypatch, capsys):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    pd.DataFrame({"value": list(range(50))}).to_csv(reference, index=False)
    pd.DataFrame({"value": list(range(100, 150))}).to_csv(current, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["framevitals", "compare", str(reference), str(current)],
    )
    assert main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "compare",
            str(reference),
            str(current),
            "--fail-on",
            "moderate",
            "--format",
            "terminal",
        ],
    )
    assert main() == 1
    output = capsys.readouterr().out
    assert "FrameVitals drift" in output
    assert "Severity" in output
