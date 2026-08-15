from pathlib import Path
from uuid import uuid4

import pandas as pd

from framevitals.security import (
    make_safe_filename,
    validate_file,
)


UPLOAD_DIR = Path("uploads")


def save_uploaded_file(uploaded_file):
    """
    Save a web-uploaded dataset and return its generated dataset ID,
    saved path, and sanitized original filename.
    """

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_file(uploaded_file.filename)

    original_filename = make_safe_filename(
        uploaded_file.filename
    )

    suffix = Path(
        original_filename
    ).suffix.lower()

    dataset_id = uuid4().hex[:12]

    saved_path = (
        UPLOAD_DIR
        / f"{dataset_id}{suffix}"
    )

    uploaded_file.save(saved_path)

    return (
        dataset_id,
        saved_path,
        original_filename,
    )


def _read_csv_tolerant(
    file_path: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """
    Read CSV/TSV files while tolerating common encodings.
    """

    path = Path(file_path)

    for encoding in (
        "utf-8-sig",
        "windows-1252",
        "latin-1",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                **kwargs,
            )
        except (
            UnicodeDecodeError,
            UnicodeError,
        ):
            continue

    return pd.read_csv(
        path,
        encoding="latin-1",
        encoding_errors="replace",
        **kwargs,
    )


def load_dataset(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load a supported tabular dataset.

    Supported formats:
    CSV, TSV, XLSX, XLS, and JSON.
    """

    path = Path(file_path)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _read_csv_tolerant(path)

    if suffix == ".tsv":
        return _read_csv_tolerant(
            path,
            sep="\t",
        )

    if suffix in {
        ".xlsx",
        ".xls",
    }:
        return pd.read_excel(path)

    if suffix == ".json":
        try:
            return pd.read_json(
                path,
                encoding="utf-8-sig",
            )
        except (
            UnicodeDecodeError,
            UnicodeError,
        ):
            return pd.read_json(
                path,
                encoding="latin-1",
            )

    raise ValueError(
        f"Unsupported file format: {suffix}"
    )
