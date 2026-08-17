"""Focused FrameVitals analysis entry points.

This module intentionally does not import the full analysis pipeline. Public
calls such as ``fv.profile`` and ``fv.statistics`` execute only the work the
caller requested. Individual analysis implementations are imported lazily too,
so a profile-only call does not load sklearn/statsmodels/deep-stat modules.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from framevitals.provenance import (
    execution_provenance,
    load_fully_materializes,
    normalize_execution,
)
from framevitals.result import DiagnosticResult
from framevitals.sources import StreamingDatasetSource, resolve_source


DataInput = Any


def _load(data: DataInput, *, label: str = "Dataset") -> tuple[pd.DataFrame, str]:
    try:
        source = resolve_source(data)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        if label == "Dataset":
            raise
        raise type(exc)(str(exc).replace("Dataset", label, 1)) from exc

    metadata = source.inspect()
    dataframe = source.load()
    return dataframe, metadata.name


def _named(
    payload: dict[str, Any],
    source_name: str,
    *,
    diagnostic: str,
) -> DiagnosticResult:
    return DiagnosticResult(
        {"dataset_name": source_name, **payload},
        diagnostic=diagnostic,
    )


def profile(data: DataInput) -> DiagnosticResult:
    """Profile a dataset, streaming Arrow-capable file sources when available."""
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import build_streaming_profile

        return _named(
            build_streaming_profile(source),
            metadata.name,
            diagnostic="profile",
        )

    from framevitals.profiler import build_profile

    dataframe = source.load()
    return _named(
        build_profile(dataframe),
        metadata.name,
        diagnostic="profile",
    )


def roles(data: DataInput) -> DiagnosticResult:
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import build_streaming_profile
        from framevitals.streaming_roles import infer_streaming_column_roles

        dataset_profile, sample = build_streaming_profile(
            source,
            sample_rows=5_000,
            return_sample=True,
        )
        payload = infer_streaming_column_roles(sample, profile=dataset_profile)
        return _named(payload, metadata.name, diagnostic="roles")

    from framevitals.column_roles import infer_column_roles, summarize_roles

    dataframe = source.load()
    column_roles = infer_column_roles(dataframe)
    return _named(
        {
            "columns": column_roles,
            "summary": summarize_roles(column_roles),
        },
        metadata.name,
        diagnostic="roles",
    )


def health(data: DataInput) -> DiagnosticResult:
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.health_score import calculate_health_score_from_profile_sample
        from framevitals.streaming_profile import build_streaming_profile

        dataset_profile, sample = build_streaming_profile(source, return_sample=True)
        payload = calculate_health_score_from_profile_sample(dataset_profile, sample)
        return _named(payload, metadata.name, diagnostic="health")

    from framevitals.health_score import calculate_health_score
    from framevitals.profiler import build_profile

    dataframe = source.load()
    dataset_profile = build_profile(dataframe)
    return _named(
        calculate_health_score(dataframe, dataset_profile),
        metadata.name,
        diagnostic="health",
    )


def ml_readiness(data: DataInput) -> DiagnosticResult:
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.ml_readiness import calculate_ml_readiness_from_profile
        from framevitals.streaming_profile import build_streaming_profile

        dataset_profile = build_streaming_profile(source)
        return _named(
            calculate_ml_readiness_from_profile(dataset_profile),
            metadata.name,
            diagnostic="ml_readiness",
        )

    from framevitals.ml_readiness import calculate_ml_readiness
    from framevitals.profiler import build_profile

    dataframe = source.load()
    dataset_profile = build_profile(dataframe)
    return _named(
        calculate_ml_readiness(dataframe, profile=dataset_profile),
        metadata.name,
        diagnostic="ml_readiness",
    )


def quality(
    data: DataInput,
    *,
    max_sample_rows: int = 5_000,
    max_columns: int = 100,
    max_missingness_columns: int = 25,
) -> DiagnosticResult:
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import build_streaming_profile
        from framevitals.streaming_quality import run_streaming_quality_diagnostics

        dataset_profile, sample = build_streaming_profile(
            source,
            sample_rows=max_sample_rows,
            return_sample=True,
        )
        payload = run_streaming_quality_diagnostics(
            sample,
            profile=dataset_profile,
            source_rows=int(metadata.rows or len(sample)),
            source_columns=int(metadata.columns or len(sample.columns)),
            max_sample_rows=max_sample_rows,
            max_columns=max_columns,
            max_missingness_columns=max_missingness_columns,
        )
        return _named(payload, metadata.name, diagnostic="quality")

    from framevitals.column_roles import infer_column_roles
    from framevitals.profiler import build_profile
    from framevitals.quality_diagnostics import run_quality_diagnostics

    dataframe = source.load()
    dataset_profile = build_profile(dataframe)
    column_roles = infer_column_roles(dataframe)
    payload = run_quality_diagnostics(
        dataframe,
        profile=dataset_profile,
        column_roles=column_roles,
        max_sample_rows=max_sample_rows,
        max_columns=max_columns,
        max_missingness_columns=max_missingness_columns,
    )
    return _named(payload, metadata.name, diagnostic="quality")


def statistics(
    data: DataInput,
    *,
    max_pairs: int = 20,
    mode: str = "standard",
) -> DiagnosticResult:
    """Run deep statistics through the same large-data budget as full analysis."""
    from framevitals.budgeted_analysis import run_budgeted_deep_statistics
    from framevitals.execution import derive_execution_budget

    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import sample_streaming_source

        if metadata.rows is None or metadata.columns is None:
            raise ValueError("Streaming statistics require source shape metadata.")

        source_rows = int(metadata.rows)
        source_columns = int(metadata.columns)
        budget = derive_execution_budget(source_rows, source_columns, mode=mode)
        sample_limit = max(
            20,
            min(
                budget.deep_statistics_sample_rows,
                budget.bootstrap_sample_rows,
            ),
        )
        sample = sample_streaming_source(source, sample_rows=sample_limit)
        payload = run_budgeted_deep_statistics(
            sample,
            budget=budget,
            max_pairs=max_pairs,
        )
        execution = dict(payload.get("execution", {}))
        sampled = len(sample) < source_rows
        execution.update({
            "sampled": sampled,
            "source_rows": source_rows,
            "source_columns": source_columns,
            "sample_rows": int(len(sample)),
            "strategy": (
                "streaming_stratified_jitter_global_rows"
                if sampled
                else "full_stream_via_batches"
            ),
            "full_materialization": False,
            "reason": (
                "Deep statistics ran on a bounded deterministic stratified-jitter sample "
                "selected directly from the streaming source."
                if sampled
                else "The streaming source fits within the deep-statistics execution budget."
            ),
        })
        payload["execution"] = normalize_execution(
            execution,
            method="bounded_deep_statistics",
            full_materialization=False,
            source=metadata.to_dict(),
        )
        payload["source"] = metadata.to_dict()
        return _named(payload, metadata.name, diagnostic="statistics")

    dataframe = source.load()
    budget = derive_execution_budget(
        len(dataframe),
        len(dataframe.columns),
        mode=mode,
    )
    payload = run_budgeted_deep_statistics(
        dataframe,
        budget=budget,
        max_pairs=max_pairs,
    )
    payload["execution"] = normalize_execution(
        payload.get("execution", {}),
        method="bounded_deep_statistics",
        full_materialization=load_fully_materializes(metadata),
        source=metadata.to_dict(),
    )
    payload["source"] = metadata.to_dict()
    return _named(payload, metadata.name, diagnostic="statistics")


def anomalies(
    data: DataInput,
    *,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
    mode: str = "standard",
) -> DiagnosticResult:
    """Run anomaly diagnostics with bounded covariance/neighbor work."""
    from framevitals.budgeted_analysis import run_budgeted_anomalies
    from framevitals.execution import derive_execution_budget

    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import (
            numeric_columns_for_streaming_source,
            sample_streaming_source,
        )

        if metadata.rows is None or metadata.columns is None:
            raise ValueError("Streaming anomaly analysis requires source shape metadata.")
        source_rows = int(metadata.rows)
        source_columns = int(metadata.columns)
        budget = derive_execution_budget(source_rows, source_columns, mode=mode)
        numeric_columns = numeric_columns_for_streaming_source(source)
        if not numeric_columns:
            execution = execution_provenance(
                "bounded_anomaly_detection",
                full_materialization=False,
                source=metadata.to_dict(),
                sampled=False,
                source_rows=source_rows,
                source_columns=source_columns,
                sample_rows=0,
                strategy="streaming_schema_only",
                scope="bounded_anomaly_detection",
                extra={"projected_columns": 0},
            )
            return _named(
                {
                    "available": False,
                    "reason": "No numeric columns available for anomaly analysis.",
                    "execution": execution,
                    "source": metadata.to_dict(),
                },
                metadata.name,
                diagnostic="anomalies",
            )

        sample_limit = max(100, int(budget.anomaly_sample_rows))
        sample = sample_streaming_source(
            source,
            sample_rows=sample_limit,
            columns=numeric_columns,
        )
        payload = run_budgeted_anomalies(
            sample,
            budget=budget,
            contamination=contamination,
            threshold=threshold,
            max_columns=max_columns,
            top_k=top_k,
        )
        execution = dict(payload.get("execution", {}))
        sampled = len(sample) < source_rows
        execution.update({
            "sampled": sampled,
            "source_rows": source_rows,
            "source_columns": source_columns,
            "projected_columns": int(len(numeric_columns)),
            "sample_rows": int(len(sample)),
            "strategy": (
                "streaming_stratified_jitter_numeric_projection"
                if sampled
                else "full_stream_numeric_projection"
            ),
            "full_materialization": False,
            "reason": (
                "Anomaly diagnostics ran on a bounded deterministic stratified-jitter "
                "numeric projection selected directly from the streaming source."
                if sampled
                else "The streaming source fits within the anomaly execution budget."
            ),
        })
        payload["execution"] = normalize_execution(
            execution,
            method="bounded_anomaly_detection",
            full_materialization=False,
            source=metadata.to_dict(),
        )
        payload["source"] = metadata.to_dict()
        return _named(payload, metadata.name, diagnostic="anomalies")

    dataframe = source.load()
    budget = derive_execution_budget(
        len(dataframe),
        len(dataframe.columns),
        mode=mode,
    )
    payload = run_budgeted_anomalies(
        dataframe,
        budget=budget,
        contamination=contamination,
        threshold=threshold,
        max_columns=max_columns,
        top_k=top_k,
    )
    payload["execution"] = normalize_execution(
        payload.get("execution", {}),
        method="bounded_anomaly_detection",
        full_materialization=load_fully_materializes(metadata),
        source=metadata.to_dict(),
    )
    payload["source"] = metadata.to_dict()
    return _named(payload, metadata.name, diagnostic="anomalies")


def relationships(
    data: DataInput,
    *,
    max_sample_rows: int = 512,
    projections: int = 64,
    min_abs_correlation: float = 0.80,
    max_candidate_pairs: int = 250_000,
    max_edges_returned: int = 5_000,
) -> DiagnosticResult:
    from framevitals.relationship_graph import build_numeric_relationship_graph

    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import (
            numeric_columns_for_streaming_source,
            sample_streaming_source,
        )

        numeric_columns = numeric_columns_for_streaming_source(source)
        sample = sample_streaming_source(
            source,
            sample_rows=max_sample_rows,
            columns=numeric_columns,
        )
        payload = build_numeric_relationship_graph(
            sample,
            max_sample_rows=max_sample_rows,
            projections=projections,
            min_abs_correlation=min_abs_correlation,
            max_candidate_pairs=max_candidate_pairs,
            max_edges_returned=max_edges_returned,
        )
        source_rows = int(metadata.rows or len(sample))
        source_columns = int(metadata.columns or len(sample.columns))
        sampled = source_rows > len(sample)
        strategy = (
            "streaming_stratified_jitter_global_rows"
            if sampled
            else "full_stream_via_batches"
        )
        sample_metadata = payload.setdefault("sample", {})
        sample_metadata.update({
            "source_rows": source_rows,
            "sample_rows": int(len(sample)),
            "sampled": sampled,
            "full_materialization": False,
            "strategy": strategy,
        })
        payload["source"] = metadata.to_dict()
        payload["streaming_source"] = True
        payload["full_materialization"] = False
        payload["execution"] = execution_provenance(
            "bounded_relationship_graph",
            full_materialization=False,
            source=metadata.to_dict(),
            sampled=sampled,
            source_rows=source_rows,
            source_columns=source_columns,
            sample_rows=int(len(sample)),
            strategy=strategy,
            components={
                "numeric_projection": "schema_exact",
                "relationship_candidates": (
                    "bounded_row_sample" if sampled else "full_input"
                ),
            },
            extra={"projected_columns": int(len(numeric_columns))},
        )
        return _named(payload, metadata.name, diagnostic="relationships")

    dataframe = source.load()
    payload = build_numeric_relationship_graph(
        dataframe,
        max_sample_rows=max_sample_rows,
        projections=projections,
        min_abs_correlation=min_abs_correlation,
        max_candidate_pairs=max_candidate_pairs,
        max_edges_returned=max_edges_returned,
    )
    sample_metadata = payload.get("sample", {})
    if not isinstance(sample_metadata, dict):
        sample_metadata = {}
    sampled = bool(sample_metadata.get("sampled", False))
    sample_rows = int(sample_metadata.get("sample_rows", len(dataframe)))
    strategy = str(sample_metadata.get("strategy", "full_input"))
    payload["source"] = metadata.to_dict()
    payload["full_materialization"] = load_fully_materializes(metadata)
    payload["execution"] = execution_provenance(
        "bounded_relationship_graph",
        full_materialization=load_fully_materializes(metadata),
        source=metadata.to_dict(),
        sampled=sampled,
        source_rows=int(len(dataframe)),
        source_columns=int(len(dataframe.columns)),
        sample_rows=sample_rows,
        strategy=strategy,
        components={
            "relationship_candidates": "bounded_row_sample" if sampled else "full_input",
        },
    )
    return _named(payload, metadata.name, diagnostic="relationships")


def target_analysis(
    data: DataInput,
    *,
    target: str,
) -> DiagnosticResult:
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_target import run_streaming_target_analysis

        payload = run_streaming_target_analysis(source, target=target)
        return _named(payload, metadata.name, diagnostic="target_analysis")

    from framevitals.column_roles import infer_column_roles
    from framevitals.target_intelligence import run_target_intelligence

    dataframe = source.load()
    if target not in dataframe.columns:
        raise ValueError(f"Target column not found: {target}")
    column_roles = infer_column_roles(dataframe)
    payload = run_target_intelligence(
        dataframe,
        target_column=target,
        column_roles=column_roles,
    )
    return _named(payload, metadata.name, diagnostic="target_analysis")
