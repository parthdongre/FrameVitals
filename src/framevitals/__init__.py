"""FrameVitals public package interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from framevitals.config import AnalysisConfig, available_modules
from framevitals.planning import AnalysisPlan
from framevitals.quality_results import DriftResult, GateResult, ValidationResult
from framevitals.result import AnalysisResult, ColumnResult
from framevitals.snapshots import (
    AnalysisSnapshot,
    compare_snapshots,
    create_snapshot,
    load_snapshot,
)

if TYPE_CHECKING:
    from framevitals.cleaning_plan import CleaningPlan

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily expose optional/heavier public types.

    ``CleaningPlan`` depends on pandas/NumPy. Delaying that import preserves the
    package's lightweight top-level import boundary and built-wheel smoke test
    while keeping ``framevitals.CleaningPlan`` available to normal users.
    """
    if name == "CleaningPlan":
        from framevitals.cleaning_plan import CleaningPlan

        return CleaningPlan
    raise AttributeError(name)


def profile(data: Any) -> dict[str, Any]:
    """Profile shape, dtypes, missingness, cardinality, and basic summaries."""
    from framevitals.focused import profile as _profile

    return _profile(data)


def roles(data: Any) -> dict[str, Any]:
    """Infer semantic and structural roles for dataset columns."""
    from framevitals.focused import roles as _roles

    return _roles(data)


def health(data: Any) -> dict[str, Any]:
    """Calculate only the FrameVitals data-health score."""
    from framevitals.focused import health as _health

    return _health(data)


def ml_readiness(data: Any) -> dict[str, Any]:
    """Calculate only ML-readiness diagnostics."""
    from framevitals.focused import ml_readiness as _ml_readiness

    return _ml_readiness(data)


def quality(
    data: Any,
    *,
    max_sample_rows: int = 5_000,
    max_columns: int = 100,
    max_missingness_columns: int = 25,
) -> dict[str, Any]:
    """Run practical deterministic data-quality diagnostics only."""
    from framevitals.focused import quality as _quality

    return _quality(
        data,
        max_sample_rows=max_sample_rows,
        max_columns=max_columns,
        max_missingness_columns=max_missingness_columns,
    )


def statistics(
    data: Any,
    *,
    max_pairs: int = 20,
    mode: str = "standard",
) -> dict[str, Any]:
    """Run the deep statistical diagnostics layer with adaptive row budgets."""
    from framevitals.focused import statistics as _statistics

    return _statistics(data, max_pairs=max_pairs, mode=mode)


def anomalies(
    data: Any,
    *,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
    mode: str = "standard",
) -> dict[str, Any]:
    """Run only the bounded FrameVitals tabular anomaly ensemble."""
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
    data: Any,
    *,
    max_sample_rows: int = 512,
    projections: int = 64,
    min_abs_correlation: float = 0.80,
    max_candidate_pairs: int = 250_000,
    max_edges_returned: int = 5_000,
) -> dict[str, Any]:
    """Discover strong numeric relationships without a dense correlation matrix."""
    from framevitals.focused import relationships as _relationships

    return _relationships(
        data,
        max_sample_rows=max_sample_rows,
        projections=projections,
        min_abs_correlation=min_abs_correlation,
        max_candidate_pairs=max_candidate_pairs,
        max_edges_returned=max_edges_returned,
    )


def system_info(*, probe_gpu: bool = True) -> dict[str, Any]:
    """Inspect FrameVitals CPU/native/CUDA capabilities without installing anything."""
    from framevitals.acceleration import system_info as _system_info

    return _system_info(probe_gpu=probe_gpu)


def target_analysis(data: Any, *, target: str) -> dict[str, Any]:
    """Run target quality, leakage, association, and split diagnostics only."""
    from framevitals.focused import target_analysis as _target_analysis

    return _target_analysis(data, target=target)


def analyze(
    data: Any,
    *,
    target: str | None = None,
    mode: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: Any = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
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
        disabled_modules=disabled_modules,
    )


def plan(
    data: Any,
    *,
    target: str | None = None,
    mode: str | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: Any = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
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
        disabled_modules=disabled_modules,
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
) -> DriftResult:
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


def validate(data: Any, contract: dict[str, Any]) -> ValidationResult:
    """Validate a dataset against an inferred or explicit data contract."""
    from framevitals.api import validate as _validate

    return _validate(data, contract)


def gate(
    current: Any,
    *,
    reference: Any = None,
    contract: Any = None,
    columns: list[str] | None = None,
    max_columns: int = 30,
    drift_warn_on: str = "moderate",
    drift_fail_on: str = "severe",
    fail_on_validation_warning: bool = False,
) -> GateResult:
    """Run contract/drift checks and return one CI-friendly quality verdict."""
    from framevitals.api import gate as _gate

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
    "AnalysisConfig",
    "AnalysisPlan",
    "AnalysisResult",
    "AnalysisSnapshot",
    "CleaningPlan",
    "ColumnResult",
    "DriftResult",
    "GateResult",
    "ValidationResult",
    "profile",
    "roles",
    "health",
    "ml_readiness",
    "quality",
    "statistics",
    "anomalies",
    "relationships",
    "system_info",
    "target_analysis",
    "analyze",
    "plan",
    "plan_cleaning",
    "clean",
    "compare",
    "infer_contract",
    "validate",
    "gate",
    "available_modules",
    "create_snapshot",
    "load_snapshot",
    "compare_snapshots",
    "__version__",
]
