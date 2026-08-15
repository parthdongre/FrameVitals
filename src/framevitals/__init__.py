"""FrameVitals public package interface."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0.dev0"


def analyze(
    data: Any,
    *,
    target: str | None = None,
    mode: str = "standard",
    artifacts: bool = False,
) -> dict[str, Any]:
    """Analyze a tabular dataset from a DataFrame or supported file path."""
    from framevitals.api import analyze as _analyze

    return _analyze(
        data,
        target=target,
        mode=mode,
        artifacts=artifacts,
    )


def compare(
    reference: Any,
    current: Any,
    *,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> dict[str, Any]:
    """Compare two datasets and return a structured drift report."""
    from framevitals.api import compare as _compare

    return _compare(
        reference,
        current,
        columns=columns,
        max_columns=max_columns,
    )


def infer_contract(
    reference: Any,
    *,
    missing_tolerance: float = 0.05,
    duplicate_tolerance: float = 0.02,
    max_allowed_values: int = 20,
) -> dict[str, Any]:
    """Infer a conservative data-health contract from a reference dataset."""
    from framevitals.api import infer_contract as _infer_contract

    return _infer_contract(
        reference,
        missing_tolerance=missing_tolerance,
        duplicate_tolerance=duplicate_tolerance,
        max_allowed_values=max_allowed_values,
    )


def validate(data: Any, contract: Any) -> dict[str, Any]:
    """Validate a dataset against a FrameVitals contract."""
    from framevitals.api import validate as _validate

    return _validate(data, contract)


__all__ = [
    "analyze",
    "compare",
    "infer_contract",
    "validate",
    "__version__",
]
