import pytest

import framevitals


def test_package_version():
    assert framevitals.__version__ == "0.1.0.dev0"


def test_analyze_is_public():
    assert callable(framevitals.analyze)


def test_missing_dataset():
    with pytest.raises(FileNotFoundError):
        framevitals.analyze(
            "this_file_does_not_exist.csv"
        )


def test_invalid_analysis_mode(tmp_path):
    dataset = tmp_path / "sample.csv"

    dataset.write_text(
        "age,income\n"
        "20,30000\n"
        "30,50000\n"
    )

    with pytest.raises(ValueError, match="Invalid analysis mode"):
        framevitals.analyze(
            dataset,
            mode="invalid-mode",
        )
