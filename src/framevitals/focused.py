"""Focused FrameVitals analysis entry points.

This module intentionally does not import the full analysis pipeline. Public
calls such as ``fv.profile`` and ``fv.statistics`` execute only the work the
caller requested. Individual analysis implementations are imported lazily too,
so a profile-only call does not load sklearn/statsmodels/deep-stat modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from framevitals.sources import StreamingDatasetSource, resolve_source


DataInput = str | Path | pd.DataFrame


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


def _named(payload: dict[str, Any], source_name: str) -> dict[str, Any]:
    return {"dataset_name": source_name, **payload}


def profile(data: DataInput) -> dict[str, Any]:
    """Profile a dataset, streaming Arrow-capable file sources when available."""
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.streaming_profile import build_streaming_profile

        return _named(build_streaming_profile(source), metadata.name)

    from framevitals.profiler import build_profile

    dataframe = source.load()
    return _named(build_profile(dataframe), metadata.name)


def roles(data: DataInput) -> dict[str, Any]:
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
        return _named(payload, metadata.name)

    from framevitals.column_roles import infer_column_roles, summarize_roles

    dataframe = source.load()
    column_roles = infer_column_roles(dataframe)
    return {
        "dataset_name": metadata.name,
        "columns": column_roles,
        "summary": summarize_roles(column_roles),
    }


def health(data: DataInput) -> dict[str, Any]:
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.health_score import calculate_health_score_from_profile_sample
        from framevitals.streaming_profile import build_streaming_profile

        dataset_profile, sample = build_streaming_profile(source, return_sample=True)
        payload = calculate_health_score_from_profile_sample(dataset_profile, sample)
        return _named(payload, metadata.name)

    from framevitals.health_score import calculate_health_score
    from framevitals.profiler import build_profile

    dataframe = source.load()
    dataset_profile = build_profile(dataframe)
    return _named(calculate_health_score(dataframe, dataset_profile), metadata.name)


def ml_readiness(data: DataInput) -> dict[str, Any]:
    source = resolve_source(data)
    metadata = source.inspect()
    if metadata.supports_streaming and isinstance(source, StreamingDatasetSource):
        from framevitals.ml_readiness import calculate_ml_readiness_from_profile
        from framevitals.streaming_profile import build_streaming_profile

        dataset_profile = build_streaming_profile(source)
        return _named(calculate_ml_readiness_from_profile(dataset_profile), metadata.name)

    from framevitals.ml_readiness import calculate_ml_readiness
    from framevitals.profiler import build_profile

    dataframe = source.load()
    dataset_profile = build_profile(dataframe)
    return _named(
        calculate_ml_readiness(dataframe, profile=dataset_profile),
        metadata.name,
    )


def quality(
    data: DataInput,
    *,
    max_sample_rows: int = 5_000,
    max_columns: int = 100,
    max_missingness_columns: int = 25,
) -> dict[str, Any]:
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
        return _named(payload, metadata.name)

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
    return _named(payload, metadata.name)


def statistics(
    data: DataInput,
    *,
    max_pairs: int = 20,
    mode: str = "standard",
) -> dict[str, Any]:
    """Run deep statistics through the same large-data budget as full analysis."""
    from framevitals.budgeted_analysis import run_budgeted_deep_statistics
    from framevitals.execution import derive_execution_budget

    dataframe, source_name = _load(data)
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
    return _named(payload, source_name)


def anomalies(
    data: DataInput,
    *,
    contamination: float = 0.05,
    threshold: float = 0.6,
    max_columns: int = 30,
    top_k: int = 25,
    mode: str = "standard",
) -> dict[str, Any]:
    """Run anomaly diagnostics with bounded covariance/neighbor work."""
    from framevitals.budgeted_analysis import run_budgeted_anomalies
    from framevitals.execution import derive_execution_budget

    dataframe, source_name = _load(data)
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
    return _named(payload, source_name)


def relationships(
    data: DataInput,
    *,
    max_sample_rows: int = 512,
    projections: int = 64,
    min_abs_correlation: float = 0.80,
    max_candidate_pairs: int = 250_000,
    max_edges_returned: int = 5_000,
) -> dict[str, Any]:
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
        sample_metadata = payload.setdefault("sample", {})
        sample_metadata.update({
            "source_rows": int(metadata.rows or len(sample)),
            "sample_rows": int(len(sample)),
            "sampled": bool((metadata.rows or len(sample)) > len(sample)),
            "full_materialization": False,
            "strategy": "streaming_evenly_spaced_global_rows",
        })
        payload["source"] = metadata.to_dict()
        payload["streaming_source"] = True
        payload["full_materialization"] = False
        return _named(payload, metadata.name)

    dataframe = source.load()
    payload = build_numeric_relationship_graph(
        dataframe,
        max_sample_rows=max_sample_rows,
        projections=projections,
        min_abs_correlation=min_abs_correlation,
        max_candidate_pairs=max_candidate_pairs,
        max_edges_returned=max_edges_returned,
    )
    return _named(payload, metadata.name)


def target_analysis(
    data: DataInput,
    *,
    target: str,
) -> dict[str, Any]:
    from framevitals.column_roles import infer_column_roles
    from framevitals.target_intelligence import run_target_intelligence

    dataframe, source_name = _load(data)
    if target not in dataframe.columns:
        raise ValueError(f"Target column not found: {target}")
    column_roles = infer_column_roles(dataframe)
    payload = run_target_intelligence(
        dataframe,
        target_column=target,
        column_roles=column_roles,
    )
    return _named(payload, source_name)
