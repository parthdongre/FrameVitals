from pathlib import Path
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".json"}

def validate_file(filename: str):
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Allowed formats are CSV, XLSX, XLS, TSV, and JSON."
        )

def make_safe_filename(filename: str):
    return secure_filename(filename)

def sanitize_csv_value(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value
