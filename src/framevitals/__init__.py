"""FrameVitals public package interface."""

from __future__ import annotations

from typing import Any

from framevitals.result import AnalysisResult, ColumnResult

__version__ = "0.1.0"


def analyze(
    data: Any,
    *,
    target: str | None = None,
    mode: str = "standard",
    artifacts: bool = False,
) -> AnalysisResult:
    """Analyze a tabular dataset from a DataFrame or supported file path.

    The analytics implementation is imported lazily so ``import framevitals``
    and ``framevitals --version`` do not initialize the complete analytics
    stack. The lightweight result types remain available at package level.
    """
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


def infer_contract(data: Any) -> dict[str, Any]:
    """Infer a reusable data contract from a reference dataset."""
    from framevitals.api import infer_contract as _infer_contract

    return _infer_contract(data)


def validate(data: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """Validate a dataset against an inferred or explicit data contract."""
    from framevitals.api import validate as _validate

    return _validate(data, contract)


__all__ = [
    "AnalysisResult",
    "ColumnResult",
    "analyze",
    "compare",
    "infer_contract",
    "validate",
    "__version__",
]
