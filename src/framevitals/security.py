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
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def validate_file(filename: str) -> None:
    """Validate that a dataset filename has a supported extension."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Dataset filename must be a non-empty string.")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Allowed formats are CSV, XLSX, XLS, TSV, and JSON."
        )


def _safe_ascii_stem(stem: str) -> str:
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ascii_name).strip("._")
    return safe or "dataset"


def make_safe_filename(filename: str) -> str:
    """Return a conservative filesystem-safe filename.

    The extension is preserved separately from the sanitized stem so valid
    non-ASCII names such as ``数据.csv`` cannot accidentally become extensionless.
    """
    if not isinstance(filename, str) or not filename.strip():
        return "dataset"

    basename = Path(filename).name
    suffix = Path(basename).suffix.lower()
    stem = Path(basename).stem
    safe_stem = _safe_ascii_stem(stem)

    if safe_stem.upper() in _WINDOWS_DEVICE_NAMES:
        safe_stem = f"_{safe_stem}"

    return f"{safe_stem}{suffix}"


def sanitize_csv_value(value):
    """Prevent spreadsheet formula injection in exported CSV string cells."""
    if not isinstance(value, str):
        return value

    stripped = value.lstrip()
    if stripped.startswith(_FORMULA_PREFIXES) or value.startswith(("\t", "\r")):
        return "'" + value
    return value
