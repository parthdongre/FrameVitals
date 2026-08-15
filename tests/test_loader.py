import pandas as pd
import pytest

from framevitals.loader import load_dataset
from framevitals.security import validate_file


def test_load_csv(tmp_path):
    dataset = tmp_path / "sample.csv"

    dataset.write_text(
        "name,score\n"
        "Alice,90\n"
        "Bob,85\n"
    )

    df = load_dataset(dataset)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == [
        "name",
        "score",
    ]


def test_validate_supported_file():
    validate_file("dataset.csv")
    validate_file("dataset.xlsx")
    validate_file("dataset.json")


def test_validate_unsupported_file():
    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        validate_file("malware.exe")


def test_loader_rejects_unknown_format(
    tmp_path,
):
    dataset = tmp_path / "dataset.xyz"

    dataset.write_text("data")

    with pytest.raises(
        ValueError,
        match="Unsupported file format",
    ):
        load_dataset(dataset)
