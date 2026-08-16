"""FrameVitals public package interface."""

from __future__ import annotations

from typing import Any

from framevitals.config import AnalysisConfig
from framevitals.planning import AnalysisPlan
from framevitals.result import AnalysisResult, ColumnResult

__version__ = "0.1.0"


def analyze(
    data: Any,
    *,
    target: str | None = None,
    mode: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: Any = None,
) -> AnalysisResult:
    """Analyze a tabular dataset from a DataFrame or supported file path."""
    from framevitals.api import analyze as _analyze

    return _analyze(
        data,
        target=target,
        mode=mode,
        artifacts=artifacts,
        workers=workers,
        preset=preset,
        config=config,
    )


def plan(
    data: Any,
    *,
    target: str | None = None,
    mode: str | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: Any = None,
) -> AnalysisPlan:
    """Preview applicable analyses without running heavy analysis stages."""
    from framevitals.api import plan as _plan

    return _plan(
        data,
        target=target,
        mode=mode,
        workers=workers,
        preset=preset,
        config=config,
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
    "AnalysisConfig",
    "AnalysisPlan",
    "AnalysisResult",
    "ColumnResult",
    "analyze",
    "plan",
    "compare",
    "infer_contract",
    "validate",
    "__version__",
]
