import pytest

from framevitals.security import make_safe_filename, sanitize_csv_value, validate_file


def test_non_ascii_upload_name_preserves_supported_extension():
    validate_file("数据.csv")
    assert make_safe_filename("数据.csv") == "dataset.csv"


def test_safe_filename_preserves_extension_after_path_stripping():
    assert make_safe_filename("../../CON.csv") == "_CON.csv"


def test_validate_file_rejects_empty_filename():
    with pytest.raises(ValueError, match="non-empty"):
        validate_file("")


def test_csv_formula_sanitizer_handles_leading_whitespace_and_control_chars():
    assert sanitize_csv_value("   =SUM(A1:A2)").startswith("'")
    assert sanitize_csv_value("\t=SUM(A1:A2)").startswith("'")
    assert sanitize_csv_value("plain text") == "plain text"
