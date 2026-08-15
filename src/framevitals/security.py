from __future__ import annotations

import re
import unicodedata
from pathlib import Path


ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".tsv",
    ".json",
}

_WINDOWS_DEVICE_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_file(filename: str) -> None:
    """Validate that a dataset uses a supported file extension."""
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Allowed formats are CSV, XLSX, XLS, TSV, and JSON."
        )


def make_safe_filename(filename: str) -> str:
    """Return a conservative filesystem-safe filename using only stdlib code."""
    basename = Path(str(filename)).name
    normalized = unicodedata.normalize("NFKD", basename)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ascii_name).strip("._")

    if not safe:
        return "dataset"

    stem = Path(safe).stem.upper()
    if stem in _WINDOWS_DEVICE_NAMES:
        safe = f"_{safe}"

    return safe


def sanitize_csv_value(value):
    """Prevent spreadsheet formula injection in exported CSV files."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value

    return value
