"""Parser preflight limits; no extraction or executable deserialization."""

from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile, ZipFile

MAX_FILE_BYTES = 96 * 1024 * 1024


def validate_input(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source.name}")
    if source.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Input exceeds the 96 MiB local parsing limit.")
    suffix = source.suffix.lower()
    if suffix in {".docx", ".pptx", ".xlsx"}:
        try:
            with ZipFile(source) as archive:
                entries = archive.infolist()
                if len(entries) > 4096:
                    raise ValueError("Archive exceeds the 4,096-entry limit.")
                if sum(item.file_size for item in entries) > 128 * 1024 * 1024:
                    raise ValueError("Archive exceeds the 128 MiB decompression budget.")
                for item in entries:
                    if item.file_size > 32 * 1024 * 1024:
                        raise ValueError("Archive member exceeds the 32 MiB limit.")
                    if item.flag_bits & 1:
                        raise ValueError("Encrypted archives are unsupported.")
                    if ".." in Path(item.filename).parts or item.filename.startswith(("/", "\\")):
                        raise ValueError("Unsafe archive member path.")
        except BadZipFile as exc:
            raise ValueError("Invalid Office document archive.") from exc
    if suffix in {".graphml", ".gexf", ".xml"}:
        try:
            from defusedxml.ElementTree import parse
        except ImportError as exc:
            raise ImportError("Safe XML parsing requires framevitals[documents,graph].") from exc
        try:
            parse(source, forbid_dtd=True, forbid_entities=True, forbid_external=True)
        except Exception as exc:
            raise ValueError("Invalid or unsafe XML input.") from exc
    return source
