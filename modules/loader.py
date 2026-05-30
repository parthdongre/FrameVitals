from pathlib import Path
from uuid import uuid4
import pandas as pd
from modules.security import validate_file, make_safe_filename

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file):
    validate_file(uploaded_file.filename)
    original_filename = make_safe_filename(uploaded_file.filename)
    suffix = Path(original_filename).suffix.lower()
    dataset_id = uuid4().hex[:12]
    saved_path = UPLOAD_DIR / f"{dataset_id}{suffix}"
    uploaded_file.save(saved_path)
    return dataset_id, saved_path, original_filename


def _read_csv_tolerant(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV/TSV trying encodings in order until one succeeds.

    Many real-world files arrive as Windows-1252, Latin-1, or UTF-8-BOM.
    pandas defaults to UTF-8 and crashes on the first non-ASCII byte it
    can't decode. We try a short waterfall:

        1. utf-8-sig   — UTF-8 with optional BOM (covers most exports)
        2. windows-1252 — the "ANSI" encoding most Windows apps produce
        3. latin-1     — accepts every byte 0x00-0xFF, so never raises;
                         non-ASCII will look garbled but the file loads.
    """
    for enc in ("utf-8-sig", "windows-1252", "latin-1"):
        try:
            return pd.read_csv(file_path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Final safety net: latin-1 with errors='replace' can't fail
    return pd.read_csv(file_path, encoding="latin-1",
                       encoding_errors="replace", **kwargs)


def load_dataset(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return _read_csv_tolerant(file_path)
    if suffix == ".tsv":
        return _read_csv_tolerant(file_path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".json":
        try:
            return pd.read_json(file_path, encoding="utf-8-sig")
        except (UnicodeDecodeError, UnicodeError):
            return pd.read_json(file_path, encoding="latin-1")

    raise ValueError(f"Unsupported file format: {suffix}")
