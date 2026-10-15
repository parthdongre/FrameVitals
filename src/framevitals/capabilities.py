"""Authoritative source capabilities shared by Python, terminal and HTTP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framevitals.core.source import recognize_source
from framevitals.file_formats import format_catalog, format_for, is_available

CAPABILITIES = {
    "tabular": (
        "Profile",
        "Health",
        "Missingness",
        "Statistics",
        "Relationships",
        "Anomalies",
        "ML readiness",
        "Drift",
    ),
    "graph": (
        "Topology",
        "Connectivity",
        "Centrality",
        "Communities",
        "Shortest paths",
        "Structural findings",
    ),
    "tensor": (
        "Shape and dtype",
        "Distribution",
        "Sparsity",
        "Matrix diagnostics",
        "Numerical findings",
    ),
    "nested": ("Schema", "Type conflicts", "Depth", "Missing fields", "Structural findings"),
    "document": (
        "Text extraction",
        "Extraction coverage",
        "Structure",
        "Encoding",
        "Document comparison",
    ),
    "model": (
        "Architecture or checkpoint metadata",
        "Parameter counts",
        "Weight diagnostics",
        "Structural findings",
    ),
    "relational": ("Tables", "Keys", "Relationships", "Referential integrity", "Join risks"),
}
MODES = ("quick", "standard", "deep", "research")


def capabilities(data: Any = None) -> dict[str, Any]:
    """Discover advertised operations without parsing files or importing frameworks.

    Capability discovery describes applicability, not a guarantee every numerical
    diagnostic will run. Actual execution availability is reported in results.
    """
    if data is None:
        return {
            "schema_version": "1",
            "modes": list(MODES),
            "formats": format_catalog(),
            "sources": {key: list(value) for key, value in CAPABILITIES.items()},
            "upload_limit_bytes": 50 * 1024 * 1024,
            "privacy": "Files are uploaded to this server. AI requires explicit consent.",
        }
    if isinstance(data, (str, Path)):
        spec = format_for(data)
        if spec is None:
            raise ValueError("Unsupported file format; run framevitals formats.")
        if not is_available(spec):
            raise ImportError(f"Install framevitals[{spec.extra}] for {spec.label}.")
    descriptor = recognize_source(data).to_dict()
    kind = descriptor["kind"]
    return {
        **descriptor,
        "analyses": list(CAPABILITIES.get(kind, ())),
        "modes": list(MODES),
        "supports_focus": kind == "tabular",
        "supports_reference": kind in CAPABILITIES,
        "limitations": (
            ["Text extraction only; OCR is not performed."]
            if kind == "document"
            else ["Runtime hooks require an in-memory PyTorch model."]
            if kind == "model"
            else []
        ),
    }
