"""FrameVitals public package interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__version__ = "0.1.0.dev0"


def analyze(
    file_path: str | Path,
    *,
    target: str | None = None,
    mode: str = "standard",
) -> dict[str, Any]:
    """Analyze a tabular dataset.

    The implementation is imported lazily so ``import framevitals`` and
    ``framevitals --version`` do not initialize the complete analytics stack.
    """
    from framevitals.api import analyze as _analyze

    return _analyze(
        file_path,
        target=target,
        mode=mode,
    )


__all__ = ["analyze", "__version__"]
