"""Backward-compatible public API facade.

Historically this module contained a second eager implementation of most
FrameVitals operations. The canonical implementations now live in focused,
source-aware modules. Keeping this file as a thin lazy facade preserves imports
such as ``from framevitals.api import analyze`` without maintaining two engines
that can diverge in behavior, streaming support, or execution metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from framevitals.cleaning_plan import CleaningPlan
    from framevitals.planning import AnalysisPlan
    from framevitals.quality_results import DriftResult, GateResult, ValidationResult
    from framevitals.result import AnalysisResult


DataInput = Any


def profile(data: DataInput) -> dict[str, Any]:
    """Profile a dataset through the canonical focused execution path."""
    from framevitals.focused import profile as _profile

    return _profile(data)


def roles(data: DataInput) -> dict[str, Any]:
    """Infer column roles through the canonical focused execution path."""
    from framevitals.focused import roles as _roles

    return _roles(data)


def health(data: DataInput) -> dict[str, Any]:
    """Calculate dataset health through the canonical focused execution path."""
    from framevitals.focused import health as _health

    return _health(data)


def ml_readiness(data: DataInput) -> dict[str, Any]:
    """Calculate ML readiness through the canonical focused execution path."""
    from framevitals.focused import ml_readiness as _ml_readiness

    return _ml_readiness(data)


def quality(
    data: DataInput,
    *,
    max_sample_rows: int = 5_000,
    max_columns: int = 100,
    max_missingness_columns: int = 25,
) -> dict[str, Any]:
    """Run deterministic quality diagnostics through the focused engine."""
    from framevitals.focused import quality as _quality

    return _quality(
        data,
        max_sample_rows=max_sample_rows,
        max_columns=max_columns,
        max_missingness_columns=max_missingness_columns,
    )


def statistics(
    data: DataInput,
    *,
    max_pairs: int = 20,
    mode: str = "standard",
) -> dict[str, Any]:
    """Run bounded deep statistics through the focused engine."""
    from framevitals.focused import statistics as _statistics

    return _statistics(data, max_pairs=max_pairs, mode=mode)


def anomalies(
    data: DataInput,
    *,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
    mode: str = "standard",
) -> dict[str, Any]:
    """Run bounded anomaly diagnostics through the focused engine."""
    from framevitals.focused import anomalies as _anomalies

    return _anomalies(
        data,
        contamination=contamination,
        threshold=threshold,
        max_columns=max_columns,
        top_k=top_k,
        mode=mode,
    )


def relationships(
    data: DataInput,
    *,
    max_sample_rows: int = 512,
    projections: int = 64,
    min_abs_correlation: float = 0.80,
    max_candidate_pairs: int = 250_000,
    max_edges_returned: int = 5_000,
) -> dict[str, Any]:
    """Discover strong numeric relationships through the focused engine."""
    from framevitals.focused import relationships as _relationships

    return _relationships(
        data,
        max_sample_rows=max_sample_rows,
        projections=projections,
        min_abs_correlation=min_abs_correlation,
        max_candidate_pairs=max_candidate_pairs,
        max_edges_returned=max_edges_returned,
    )


def target_analysis(data: DataInput, *, target: str) -> dict[str, Any]:
    """Run target diagnostics through the focused source-aware engine."""
    from framevitals.focused import target_analysis as _target_analysis

    return _target_analysis(data, target=target)


def analyze(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: Any = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
) -> AnalysisResult:
    """Analyze a dataset through the canonical source-aware dispatcher."""
    from framevitals.analysis_api import analyze as _analyze

    return _analyze(
        data,
        target=target,
        mode=mode,
        artifacts=artifacts,
        workers=workers,
        preset=preset,
        config=config,
        disabled_modules=disabled_modules,
    )


def plan(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: Any = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
) -> AnalysisPlan:
    """Preview analysis execution through the canonical planning API."""
    from framevitals.planning_api import plan as _plan

    return _plan(
        data,
        target=target,
        mode=mode,
        workers=workers,
        preset=preset,
        config=config,
        disabled_modules=disabled_modules,
    )


def plan_cleaning(data: DataInput) -> CleaningPlan:
    """Infer a conservative cleaning plan through the operations layer."""
    from framevitals.operations import plan_cleaning as _plan_cleaning

    return _plan_cleaning(data)


def clean(
    data: DataInput,
    *,
    plan: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return an explicitly cleaned copy through the operations layer."""
    from framevitals.operations import clean as _clean

    return _clean(data, plan=plan)


def compare(
    reference: DataInput,
    current: DataInput,
    *,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> DriftResult:
    """Compare reference/current data through the source-aware drift path."""
    from framevitals.operations import compare as _compare

    return _compare(
        reference,
        current,
        columns=columns,
        max_columns=max_columns,
    )


def infer_contract(
    data: DataInput,
    *,
    numeric_tolerance: float = 0.05,
    max_categories: int = 20,
    null_fraction_tolerance: float = 0.05,
    infer_unique: bool = True,
    min_unique_rows: int = 20,
    allow_extra_columns: bool = False,
) -> dict[str, Any]:
    """Infer a reusable data contract through the operations layer."""
    from framevitals.operations import infer_contract as _infer_contract

    return _infer_contract(
        data,
        numeric_tolerance=numeric_tolerance,
        max_categories=max_categories,
        null_fraction_tolerance=null_fraction_tolerance,
        infer_unique=infer_unique,
        min_unique_rows=min_unique_rows,
        allow_extra_columns=allow_extra_columns,
    )


def validate(data: DataInput, contract: Mapping[str, Any]) -> ValidationResult:
    """Validate a dataset exactly through the operations layer."""
    from framevitals.operations import validate as _validate

    return _validate(data, contract)


def gate(
    current: DataInput,
    *,
    reference: DataInput | None = None,
    contract: Mapping[str, Any] | None = None,
    columns: list[str] | None = None,
    max_columns: int = 30,
    drift_warn_on: str = "moderate",
    drift_fail_on: str = "severe",
    fail_on_validation_warning: bool = False,
) -> GateResult:
    """Run the canonical CI-friendly contract/drift quality gate."""
    from framevitals.operations import gate as _gate

    return _gate(
        current,
        reference=reference,
        contract=contract,
        columns=columns,
        max_columns=max_columns,
        drift_warn_on=drift_warn_on,
        drift_fail_on=drift_fail_on,
        fail_on_validation_warning=fail_on_validation_warning,
    )


__all__ = [
    "profile",
    "roles",
    "health",
    "ml_readiness",
    "quality",
    "statistics",
    "anomalies",
    "relationships",
    "target_analysis",
    "analyze",
    "plan",
    "plan_cleaning",
    "clean",
    "compare",
    "infer_contract",
    "validate",
    "gate",
]
