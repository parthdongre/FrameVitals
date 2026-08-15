import pandas as pd
import pytest

from framevitals.loader import load_dataset
from framevitals.security import make_safe_filename, sanitize_csv_value, validate_file


def test_load_csv(tmp_path):
    dataset = tmp_path / "sample.csv"
    dataset.write_text("name,score\nAlice,90\nBob,85\n")

    df = load_dataset(dataset)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["name", "score"]


def test_validate_supported_file():
    validate_file("dataset.csv")
    validate_file("dataset.xlsx")
    validate_file("dataset.json")


def test_validate_unsupported_file():
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_file("malware.exe")


def test_loader_rejects_unknown_format(tmp_path):
    dataset = tmp_path / "dataset.xyz"
    dataset.write_text("data")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset(dataset)


def test_make_safe_filename_removes_path_and_unsafe_characters():
    assert make_safe_filename("../../customers.csv") == "customers.csv"
    assert make_safe_filename("sales report?.csv") == "sales_report_.csv"


def test_make_safe_filename_avoids_windows_device_names():
    assert make_safe_filename("CON.txt") == "_CON.txt"


def test_sanitize_csv_value_blocks_formula_prefixes():
    for value in ("=1+1", "+SUM(A1:A2)", "-10+20", "@cmd"):
        assert sanitize_csv_value(value).startswith("'")

    assert sanitize_csv_value("normal") == "normal"
    assert sanitize_csv_value(42) == 42
