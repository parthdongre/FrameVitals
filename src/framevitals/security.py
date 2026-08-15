from pathlib import Path

from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".tsv",
    ".json",
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
    """Return a filesystem-safe version of a filename."""

    return secure_filename(filename)


def sanitize_csv_value(value):
    """
    Prevent spreadsheet formula injection in exported CSV files.
    """

    if isinstance(value, str) and value.startswith(
        ("=", "+", "-", "@")
    ):
        return "'" + value

    return value
