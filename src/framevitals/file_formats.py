"""Single source of truth for CLI and terminal-supported file formats.

Entries marked optional need the corresponding extra. Recognition deliberately
does not try to deserialize unsafe Python pickle formats.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormatSpec:
    extensions: tuple[str, ...]
    label: str
    category: str
    dependency: str | None = None
    extra: str | None = None
    note: str = ""


FORMATS: tuple[FormatSpec, ...] = (
    FormatSpec((".csv", ".tsv"), "Delimited data", "Tabular"),
    FormatSpec((".parquet",), "Parquet", "Tabular", "pyarrow", "arrow"),
    FormatSpec((".xlsx",), "Excel workbook", "Tabular", "openpyxl", "excel"),
    FormatSpec((".xls",), "Legacy Excel", "Tabular", "xlrd", "excel"),
    FormatSpec((".json", ".jsonl", ".ndjson"), "JSON / JSON Lines", "Nested"),
    FormatSpec((".yaml", ".yml"), "YAML", "Nested", "yaml", "documents"),
    FormatSpec((".toml",), "TOML", "Nested"),
    FormatSpec((".ipynb",), "Jupyter notebook (JSON)", "Nested"),
    FormatSpec((".npy",), "NumPy array", "Tensor"),
    FormatSpec((".graphml", ".gexf", ".gml"), "Graph file", "Graph", "networkx", "graph"),
    FormatSpec((".safetensors",), "Safetensors checkpoint", "Model"),
    FormatSpec((".onnx",), "ONNX model", "Model", "onnx", "onnx"),
    FormatSpec((".pdf",), "PDF document", "Document", "pypdf", "documents",
               "Text extraction only; image-only scans need OCR"),
    FormatSpec((".docx",), "Word document", "Document", "docx", "documents"),
    FormatSpec((".pptx",), "PowerPoint presentation", "Document", "pptx", "documents"),
    FormatSpec((".txt", ".md", ".markdown", ".rst", ".log"), "Text / Markdown", "Document"),
    FormatSpec((".html", ".htm"), "HTML document", "Document"),
    FormatSpec((".xml",), "XML document", "Document", "defusedxml", "documents"),
    FormatSpec((".py", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".tsx"),
               "Source code text", "Document"),
)
_FORMAT_BY_EXTENSION = {ext: spec for spec in FORMATS for ext in spec.extensions}


def format_for(path: str | Path) -> FormatSpec | None:
    return _FORMAT_BY_EXTENSION.get(Path(path).suffix.lower())


def supported_extensions() -> tuple[str, ...]:
    return tuple(sorted(_FORMAT_BY_EXTENSION))


def is_available(spec: FormatSpec) -> bool:
    return spec.dependency is None or find_spec(spec.dependency) is not None


def check_file_format(path: str | Path) -> FormatSpec:
    """Fail before analytics with specific file-format/dependency instructions."""
    value = Path(path).expanduser()
    spec = format_for(value)
    if spec is None:
        raise ValueError(
            f"Unsupported file type {value.suffix or '<no extension>'!r}. "
            "Run 'framevitals formats' to see supported formats. "
            "Unsafe pickle-based checkpoints (.pt/.pth/.pkl) are not auto-loaded."
        )
    if not value.is_file():
        raise FileNotFoundError(f"File does not exist: {value}")
    if not is_available(spec):
        raise ImportError(
            f"{value.suffix} support requires {spec.dependency!r}. "
            f'Install: python -m pip install "framevitals[{spec.extra}]"'
        )
    return spec


def format_catalog() -> list[dict[str, Any]]:
    return [
        {
            **asdict(spec),
            "installed": is_available(spec),
        }
        for spec in FORMATS
    ]


def render_formats() -> str:
    lines = [
        "FRAMEVITALS | Supported local files",
        "─" * 70,
        f"{'Category':<12} {'Extension(s)':<32} {'Status':<9} Description",
    ]
    for spec in FORMATS:
        extensions = ", ".join(spec.extensions)
        status = "Ready" if is_available(spec) else f"+{spec.extra}"
        lines.append(f"{spec.category:<12} {extensions:<32} {status:<9} {spec.label}")
    lines.extend([
        "",
        "Install optional formats: python -m pip install 'framevitals[documents,graph,onnx,excel,arrow]'",
        "Text-only PDF inspection does not OCR scanned pages.",
        "Untrusted pickle/model checkpoints (.pt/.pth/.pkl) are intentionally unsupported.",
        "List in the terminal dashboard: press L.",
    ])
    return "\n".join(lines)


def prepare_file_source(path: str | Path) -> Any:
    """Normalize safe serializations consistently for CLI and TUI.

    Other sources stay as paths for the existing graph/model/tabular analyzers.
    Large text-like documents are handled by the bounded document analyzer.
    """
    value = Path(path).expanduser()
    spec = check_file_format(value)
    suffix = value.suffix.lower()
    if suffix == ".npy":
        import numpy as np

        return np.load(value, mmap_mode="r", allow_pickle=False)
    if suffix in {".json", ".ipynb"}:
        import json

        if value.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("JSON exceeds the 32 MiB in-memory parsing budget.")
        with value.open(encoding="utf-8") as stream:
            return json.load(stream)
    if suffix in {".jsonl", ".ndjson"}:
        import json

        if value.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("JSON Lines exceeds the 32 MiB parsing budget.")
        with value.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) > 100_000:
            raise ValueError("JSON Lines exceeds the 100,000-record analysis budget.")
        return rows
    if suffix in {".yml", ".yaml"}:
        import yaml

        if value.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("YAML exceeds the 16 MiB parsing budget.")
        with value.open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    if suffix == ".toml":
        import tomllib

        if value.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("TOML exceeds the 16 MiB parsing budget.")
        with value.open("rb") as stream:
            return tomllib.load(stream)
    return value
