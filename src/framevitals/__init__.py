"""FrameVitals public package interface."""

from __future__ import annotations

from typing import Any

from framevitals.cleaning_plan import CleaningPlan
from framevitals.config import AnalysisConfig
from framevitals.planning import AnalysisPlan
from framevitals.result import AnalysisResult, ColumnResult
from framevitals.snapshots import (
    AnalysisSnapshot,
    compare_snapshots,
    create_snapshot,
    load_snapshot,
)

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


def plan_cleaning(data: Any) -> CleaningPlan:
    """Infer a conservative cleaning plan without changing the input data."""
    from framevitals.api import plan_cleaning as _plan_cleaning

    return _plan_cleaning(data)


def clean(data: Any, *, plan: Any = None):
    """Return an explicitly cleaned copy using a supplied or inferred plan."""
    from framevitals.api import clean as _clean

    return _clean(data, plan=plan)


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
    data: Any,
    *,
    numeric_tolerance: float = 0.05,
    max_categories: int = 20,
    null_fraction_tolerance: float = 0.05,
    infer_unique: bool = True,
    min_unique_rows: int = 20,
    allow_extra_columns: bool = False,
) -> dict[str, Any]:
    """Infer a reusable data contract from a reference dataset."""
    from framevitals.api import infer_contract as _infer_contract

    return _infer_contract(
        data,
        numeric_tolerance=numeric_tolerance,
        max_categories=max_categories,
        null_fraction_tolerance=null_fraction_tolerance,
        infer_unique=infer_unique,
        min_unique_rows=min_unique_rows,
        allow_extra_columns=allow_extra_columns,
    )


def validate(data: Any, contract: dict[str, Any]) -> dict[str, Any]:
    """Validate a dataset against an inferred or explicit data contract."""
    from framevitals.api import validate as _validate

    return _validate(data, contract)


__all__ = [
    "AnalysisConfig",
    "AnalysisPlan",
    "AnalysisResult",
    "AnalysisSnapshot",
    "CleaningPlan",
    "ColumnResult",
    "analyze",
    "plan",
    "plan_cleaning",
    "clean",
    "compare",
    "infer_contract",
    "validate",
    "create_snapshot",
    "load_snapshot",
    "compare_snapshots",
    "__version__",
]
