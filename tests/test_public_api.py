import pandas as pd
import pytest

import framevitals


def test_package_version():
    assert framevitals.__version__ == "0.1.0"


def test_analyze_is_public():
    assert callable(framevitals.analyze)


def test_compare_is_public():
    assert callable(framevitals.compare)


def test_contract_apis_are_public():
    assert callable(framevitals.infer_contract)
    assert callable(framevitals.validate)


def test_missing_dataset():
    with pytest.raises(FileNotFoundError):
        framevitals.analyze("this_file_does_not_exist.csv")


def test_invalid_analysis_mode(tmp_path):
    dataset = tmp_path / "sample.csv"
    dataset.write_text(
        "age,income\n"
        "20,30000\n"
        "30,50000\n"
    )

    with pytest.raises(ValueError, match="Invalid analysis mode"):
        framevitals.analyze(dataset, mode="invalid-mode")


def test_analyze_accepts_dataframe_without_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "income": [30000, 45000, 60000, 80000],
        "city": ["Pune", "Pune", "Mumbai", "Nashik"],
    })

    result = framevitals.analyze(df, mode="quick")

    assert result["filename"] == "<dataframe>"
    assert result["profile"]["shape"]["rows"] == 4
    assert result["artifacts_enabled"] is False
    assert result["cleaning"]["output_path"] is None
    assert not (tmp_path / "cleaned").exists()


def test_analyze_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        framevitals.analyze(pd.DataFrame(), mode="quick")


def test_compare_accepts_dataframes():
    reference = pd.DataFrame({
        "value": list(range(40)),
        "segment": ["a", "b"] * 20,
    })
    current = pd.DataFrame({
        "value": list(range(10, 50)),
        "segment": ["a"] * 30 + ["c"] * 10,
    })

    result = framevitals.compare(reference, current)

    assert result["available"] is True
    assert result["reference_name"] == "<dataframe>"
    assert result["current_name"] == "<dataframe>"
    assert result["summary"]["n_columns_compared"] == 2
    assert set(result["shared_columns"]) == {"value", "segment"}


def test_compare_validates_max_columns():
    df = pd.DataFrame({"value": range(20)})
    with pytest.raises(ValueError, match="max_columns"):
        framevitals.compare(df, df, max_columns=0)
