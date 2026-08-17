"""Target-aware diagnostics for streaming dataset sources."""

from __future__ import annotations

from typing import Any

from framevitals.column_roles import infer_column_roles
from framevitals.execution import derive_execution_budget
from framevitals.sources import StreamingDatasetSource
from framevitals.streaming_profile import sample_streaming_source
from framevitals.target_intelligence import run_target_intelligence


def _column_names(source: StreamingDatasetSource) -> list[str]:
    schema_method = getattr(source, "schema", None)
    if callable(schema_method):
        schema = schema_method()
        names = getattr(schema, "names", None)
        if names is not None:
            return [str(name) for name in names]

    first_batch = next(source.iter_batches(batch_size=1), None)
    if first_batch is None:
        return []
    schema = getattr(first_batch, "schema", None)
    names = getattr(schema, "names", None)
    if names is None:
        return []
    return [str(name) for name in names]


def run_streaming_target_analysis(
    source: StreamingDatasetSource,
    *,
    target: str,
    max_columns: int = 200,
) -> dict[str, Any]:
    """Run target diagnostics on a deterministic bounded source projection.

    The target is always retained. Feature columns are kept in source order up
    to ``max_columns - 1``. This prevents an ultra-wide source from turning a
    target-only diagnostic request into an unbounded pandas materialization.
    """
    if max_columns < 2:
        raise ValueError("max_columns must be at least 2.")

    metadata = source.inspect()
    if metadata.rows is None or metadata.columns is None:
        raise ValueError("Streaming target analysis requires source shape metadata.")
    source_rows = int(metadata.rows)
    source_columns = int(metadata.columns)

    names = _column_names(source)
    if target not in names:
        raise ValueError(f"Target column not found: {target}")

    features = [name for name in names if name != target]
    selected_features = features[: max_columns - 1]
    selected_set = set(selected_features)
    selected_columns = [
        name for name in names if name == target or name in selected_set
    ]

    budget = derive_execution_budget(
        source_rows,
        source_columns,
        mode="standard",
    )
    sample_limit = max(100, int(budget.pair_sample_rows))
    sample = sample_streaming_source(
        source,
        sample_rows=sample_limit,
        columns=selected_columns,
    )

    column_roles = infer_column_roles(sample)
    payload = run_target_intelligence(
        sample,
        target_column=target,
        column_roles=column_roles,
    )
    payload["execution"] = {
        "scope": (
            "bounded_row_and_column_sample"
            if len(sample) < source_rows or len(selected_columns) < source_columns
            else "full_source"
        ),
        "full_materialization": False,
        "source_rows": source_rows,
        "source_columns": source_columns,
        "sample_rows": int(len(sample)),
        "sampled_rows": bool(len(sample) < source_rows),
        "selected_columns": int(len(selected_columns)),
        "feature_columns_considered": int(len(selected_features)),
        "feature_columns_available": int(len(features)),
        "columns_truncated": bool(len(selected_columns) < source_columns),
        "strategy": "streaming_evenly_spaced_rows_with_deterministic_feature_projection",
        "target_always_retained": True,
    }
    payload["source"] = metadata.to_dict()
    return payload
