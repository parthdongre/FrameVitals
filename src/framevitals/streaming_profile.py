"""Streaming profile construction for Arrow-capable dataset sources.

The streaming path keeps full-row work to mergeable column state and retains
only a bounded, evenly spaced row sample for analyses that still require row
relationships (duplicate estimation and correlation). Native builds can also
consume Arrow UTF-8 buffers directly for full-file categorical sketches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from framevitals.analysis_state import NumericColumnState
from framevitals.backends import (
    create_numeric_accumulator,
    create_string_accumulator,
    numeric_state,
    resolve_numeric_backend,
)
from framevitals.profiler import _bounded_correlations, series_to_dict
from framevitals.sources import StreamingDatasetSource
from framevitals.streaming_sketches import (
    NumpyLogQuantileSketch,
    PYTHON_NUMERIC_SKETCH_CELL_BUDGET,
    should_use_full_stream_numpy_sketch,
)


STREAM_BATCH_SIZE = 65_536
STREAM_SAMPLE_ROWS = 50_000
STREAM_BATCH_CELL_BUDGET = 32_000_000
STREAM_SAMPLE_CELL_BUDGET = 6_000_000


def _width_aware_row_limit(
    requested_rows: int,
    column_count: int,
    *,
    cell_budget: int,
) -> int:
    """Clamp a row budget so rows * columns stays within a bounded cell budget."""
    if requested_rows < 1:
        raise ValueError("requested_rows must be at least 1.")
    if cell_budget < 1:
        raise ValueError("cell_budget must be at least 1.")

    width = max(int(column_count), 1)
    width_limited_rows = max(int(cell_budget) // width, 1)
    return max(1, min(int(requested_rows), width_limited_rows))


def _require_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(
            "Streaming Arrow profiling requires the optional Arrow capability. "
            'Install it with: pip install "framevitals[arrow]"'
        ) from exc
    return pa


def _column_groups(schema) -> tuple[list[str], list[str], list[str]]:
    pa = _require_pyarrow()
    numeric: list[str] = []
    categorical: list[str] = []
    dates: list[str] = []

    for field in schema:
        dtype = field.type
        if pa.types.is_integer(dtype) or pa.types.is_floating(dtype):
            numeric.append(field.name)
        elif (
            pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_dictionary(dtype)
            or pa.types.is_boolean(dtype)
            or pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
        ):
            categorical.append(field.name)
        elif (
            pa.types.is_timestamp(dtype)
            or pa.types.is_date32(dtype)
            or pa.types.is_date64(dtype)
            or pa.types.is_time32(dtype)
            or pa.types.is_time64(dtype)
            or pa.types.is_duration(dtype)
        ):
            dates.append(field.name)

    return numeric, categorical, dates


def _string_columns(schema) -> list[str]:
    pa = _require_pyarrow()
    return [
        field.name
        for field in schema
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ]


def numeric_columns_for_streaming_source(source: StreamingDatasetSource) -> list[str]:
    """Return Arrow numeric column names without reading row data."""
    schema_method = getattr(source, "schema", None)
    if not callable(schema_method):
        raise TypeError("Streaming profile sources must expose an Arrow schema().")
    numeric, _, _ = _column_groups(schema_method())
    return numeric


def _arrow_numeric_to_float64(array) -> np.ndarray:
    """Convert one bounded Arrow numeric array to a contiguous float64 buffer."""
    try:
        values = array.to_numpy(zero_copy_only=False)
        converted = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        series = array.to_pandas()
        converted = pd.to_numeric(series, errors="coerce").to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
    return np.ascontiguousarray(converted, dtype=np.float64)


def _update_native_string_accumulator(accumulator, array) -> None:
    """Feed Arrow UTF-8 buffers to Rust without constructing Python strings."""
    pa = _require_pyarrow()
    validity_buffer, offsets_buffer, data_buffer = array.buffers()
    validity = (
        np.frombuffer(validity_buffer, dtype=np.uint8)
        if validity_buffer is not None
        else None
    )
    data = (
        np.frombuffer(data_buffer, dtype=np.uint8)
        if data_buffer is not None
        else np.empty(0, dtype=np.uint8)
    )

    if pa.types.is_large_string(array.type):
        offsets = np.frombuffer(offsets_buffer, dtype=np.int64)
        accumulator.update_large_utf8(
            data,
            offsets,
            len(array),
            validity,
            int(array.offset),
        )
    else:
        offsets = np.frombuffer(offsets_buffer, dtype=np.int32)
        accumulator.update_utf8(
            data,
            offsets,
            len(array),
            validity,
            int(array.offset),
        )


def _state_from_payload(payload: dict[str, Any]) -> NumericColumnState:
    count = int(payload["count"])
    variance = payload.get("variance")
    return NumericColumnState(
        count=count,
        missing=int(payload["missing"]),
        mean=float(payload["mean"]) if count else 0.0,
        m2=(float(variance) * (count - 1) if variance is not None and count >= 2 else 0.0),
        minimum=(
            float(payload["minimum"])
            if payload.get("minimum") is not None
            else None
        ),
        maximum=(
            float(payload["maximum"])
            if payload.get("maximum") is not None
            else None
        ),
        infinite=int(payload["infinite"]),
    )


def _round_optional(value: Any, digits: int = 3):
    if value is None:
        return None
    return round(float(value), digits)


def _summary_from_native_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    quantiles = payload.get("quantiles", {})
    return {
        "count": int(payload["count"]),
        "mean": _round_optional(payload.get("mean")),
        "std": _round_optional(payload.get("std")),
        "min": _round_optional(payload.get("minimum")),
        "25%": _round_optional(quantiles.get("p25")),
        "50%": _round_optional(quantiles.get("p50")),
        "75%": _round_optional(quantiles.get("p75")),
        "max": _round_optional(payload.get("maximum")),
    }


def _summary_from_python_state(
    state: NumericColumnState,
    sample: pd.Series,
    quantile_sketch: NumpyLogQuantileSketch | None = None,
) -> dict[str, Any]:
    if quantile_sketch is not None:
        q25 = quantile_sketch.quantile(0.25)
        q50 = quantile_sketch.quantile(0.50)
        q75 = quantile_sketch.quantile(0.75)
    else:
        finite_sample = pd.to_numeric(sample, errors="coerce")
        finite_values = finite_sample.to_numpy(dtype="float64", na_value=np.nan)
        finite_sample = finite_sample[np.isfinite(finite_values)]
        quantiles = (
            finite_sample.quantile([0.25, 0.5, 0.75])
            if len(finite_sample)
            else pd.Series()
        )
        q25 = quantiles.get(0.25)
        q50 = quantiles.get(0.5)
        q75 = quantiles.get(0.75)
    return {
        "count": state.count,
        "mean": _round_optional(state.mean if state.count else None),
        "std": _round_optional(state.std),
        "min": _round_optional(state.minimum),
        "25%": _round_optional(q25),
        "50%": _round_optional(q50),
        "75%": _round_optional(q75),
        "max": _round_optional(state.maximum),
    }


def _sample_positions(rows: int, target_rows: int) -> np.ndarray:
    count = min(rows, target_rows)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == rows:
        return np.arange(rows, dtype=np.int64)
    return np.unique(np.linspace(0, rows - 1, num=count, dtype=np.int64))


def _sample_batch(batch, positions: np.ndarray, offset: int):
    pa = _require_pyarrow()
    if positions.size == 0 or batch.num_rows == 0:
        return None
    left = int(np.searchsorted(positions, offset, side="left"))
    right = int(np.searchsorted(positions, offset + batch.num_rows, side="left"))
    if right <= left:
        return None
    local = positions[left:right] - offset
    return batch.take(pa.array(local, type=pa.int64())).to_pandas()


def sample_streaming_source(
    source: StreamingDatasetSource,
    *,
    sample_rows: int,
    batch_size: int = STREAM_BATCH_SIZE,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return an evenly spaced bounded row sample without full materialization."""
    if sample_rows < 1:
        raise ValueError("sample_rows must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    metadata = source.inspect()
    rows = metadata.rows
    if rows is None:
        raise ValueError("Streaming sampling requires a source row count.")
    if rows < 1:
        raise ValueError(f"Dataset is empty: {metadata.name}")

    projected_width = len(columns) if columns is not None else int(metadata.columns or 1)
    effective_batch_size = _width_aware_row_limit(
        int(batch_size),
        projected_width,
        cell_budget=STREAM_BATCH_CELL_BUDGET,
    )
    effective_sample_rows = _width_aware_row_limit(
        int(sample_rows),
        projected_width,
        cell_budget=STREAM_SAMPLE_CELL_BUDGET,
    )

    positions = _sample_positions(int(rows), effective_sample_rows)
    frames: list[pd.DataFrame] = []
    offset = 0
    for batch in source.iter_batches(batch_size=effective_batch_size, columns=columns):
        sampled = _sample_batch(batch, positions, offset)
        if sampled is not None and not sampled.empty:
            frames.append(sampled)
        offset += int(batch.num_rows)

    if offset != rows:
        raise ValueError(
            f"Streaming source metadata reported {rows} rows but yielded {offset}."
        )

    if frames:
        sample = pd.concat(frames, ignore_index=True)
        return sample.head(len(positions))
    return pd.DataFrame(columns=list(columns or ()))


def _duplicate_summary(sample: pd.DataFrame, rows: int) -> tuple[int, dict[str, Any]]:
    if rows == len(sample):
        count = int(sample.duplicated().sum())
        return count, {
            "method": "exact_streaming_sample",
            "sampled": False,
            "sample_rows": int(len(sample)),
        }

    sample_duplicates = int(sample.duplicated().sum())
    rate = sample_duplicates / max(len(sample), 1)
    estimate = int(round(rate * rows))
    return estimate, {
        "method": "streaming_sample_estimate",
        "sampled": True,
        "sample_rows": int(len(sample)),
        "source_rows": int(rows),
        "estimated_duplicate_rate": round(float(rate), 6),
    }


def _estimated_memory_usage(sample: pd.DataFrame, rows: int) -> tuple[float, dict[str, Any]]:
    if len(sample) == 0:
        return 0.0, {"method": "unavailable", "estimated": True}
    sample_bytes = int(sample.memory_usage(index=True, deep=True).sum())
    if len(sample) == rows:
        estimated_bytes = sample_bytes
        method = "exact_from_full_stream_sample"
        estimated = False
    else:
        estimated_bytes = int(round(sample_bytes / len(sample) * rows))
        method = "scaled_from_even_row_sample"
        estimated = True
    return round(estimated_bytes / (1024 * 1024), 3), {
        "method": method,
        "estimated": estimated,
        "sample_rows": int(len(sample)),
        "source_rows": int(rows),
    }


def build_streaming_profile(
    source: StreamingDatasetSource,
    *,
    batch_size: int = STREAM_BATCH_SIZE,
    sample_rows: int = STREAM_SAMPLE_ROWS,
    return_sample: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame]:
    """Profile a streaming Arrow source without materializing all rows."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if sample_rows < 1:
        raise ValueError("sample_rows must be at least 1.")

    metadata = source.inspect()
    rows = metadata.rows
    if rows is None:
        raise ValueError("Streaming profiling currently requires a source row count.")
    if rows < 1:
        raise ValueError(f"Dataset is empty: {metadata.name}")

    schema_method = getattr(source, "schema", None)
    if not callable(schema_method):
        raise TypeError("Streaming profile sources must expose an Arrow schema().")
    schema = schema_method()
    columns = [field.name for field in schema]
    numeric_cols, categorical_cols, date_cols = _column_groups(schema)
    numeric_col_set = set(numeric_cols)
    string_cols = _string_columns(schema)
    dtypes = {field.name: str(field.type) for field in schema}

    width = max(len(columns), 1)
    effective_batch_size = _width_aware_row_limit(
        int(batch_size),
        width,
        cell_budget=STREAM_BATCH_CELL_BUDGET,
    )
    effective_sample_rows = _width_aware_row_limit(
        int(sample_rows),
        width,
        cell_budget=STREAM_SAMPLE_CELL_BUDGET,
    )
    positions = _sample_positions(int(rows), effective_sample_rows)
    sample_frames: list[pd.DataFrame] = []
    preview_frame: pd.DataFrame | None = None
    missing_counts = {column: 0 for column in columns}
    selected_backend = resolve_numeric_backend()
    use_numpy_quantile_sketches = (
        selected_backend == "numpy"
        and should_use_full_stream_numpy_sketch(int(rows), len(numeric_cols))
    )

    native_accumulators: dict[str, Any] = {}
    native_string_accumulators: dict[str, Any] = {}
    python_states = {column: NumericColumnState() for column in numeric_cols}
    numpy_quantile_sketches = (
        {column: NumpyLogQuantileSketch() for column in numeric_cols}
        if use_numpy_quantile_sketches
        else {}
    )
    if selected_backend == "rust":
        native_accumulators = {
            column: create_numeric_accumulator(stream_id=index)
            for index, column in enumerate(numeric_cols)
        }
        native_string_accumulators = {
            column: accumulator
            for column in string_cols
            if (accumulator := create_string_accumulator()) is not None
        }

    offset = 0
    batches_scanned = 0
    for batch in source.iter_batches(batch_size=effective_batch_size):
        batches_scanned += 1
        if preview_frame is None:
            preview_frame = batch.slice(0, min(15, batch.num_rows)).to_pandas()

        sampled = _sample_batch(batch, positions, offset)
        if sampled is not None and not sampled.empty:
            sample_frames.append(sampled)

        for index, column in enumerate(columns):
            array = batch.column(index)
            if column not in numeric_col_set:
                missing_counts[column] += int(array.null_count)
                string_accumulator = native_string_accumulators.get(column)
                if string_accumulator is not None:
                    _update_native_string_accumulator(string_accumulator, array)
                continue

            values = _arrow_numeric_to_float64(array)
            accumulator = native_accumulators.get(column)
            if accumulator is not None:
                accumulator.update_f64(values)
            else:
                numeric_payload = numeric_state(values, backend="numpy")
                python_states[column] = python_states[column].merge(
                    _state_from_payload(numeric_payload)
                )
                quantile_sketch = numpy_quantile_sketches.get(column)
                if quantile_sketch is not None:
                    quantile_sketch.update(values)

        offset += int(batch.num_rows)

    if offset != rows:
        raise ValueError(
            f"Streaming source metadata reported {rows} rows but yielded {offset}."
        )

    sample = (
        pd.concat(sample_frames, ignore_index=True)
        if sample_frames
        else pd.DataFrame(columns=columns)
    )
    if len(sample) > len(positions):
        sample = sample.head(len(positions))

    numeric_summary: dict[str, dict[str, Any]] = {}
    quantile_accuracy = None
    if selected_backend == "rust":
        for column in numeric_cols:
            numeric_payload = dict(native_accumulators[column].snapshot())
            numeric_summary[column] = _summary_from_native_snapshot(numeric_payload)
            missing_counts[column] = int(numeric_payload["missing"])
            quantile_accuracy = numeric_payload.get("quantiles", {}).get(
                "relative_accuracy",
                quantile_accuracy,
            )
        numeric_metadata = {
            "backend": "rust",
            "method": "native_streaming_accumulator",
            "approximate_quantiles": True,
            "quantile_relative_accuracy": quantile_accuracy,
            "quantile_source": "full_stream_sketch",
            "finite_only_moments": True,
            "columns_profiled": len(numeric_cols),
            "raw_observations_retained": False,
        }
    else:
        for column in numeric_cols:
            state = python_states[column]
            numeric_summary[column] = _summary_from_python_state(
                state,
                sample[column] if column in sample else pd.Series(dtype="float64"),
                numpy_quantile_sketches.get(column),
            )
            missing_counts[column] = state.missing
        if use_numpy_quantile_sketches:
            numeric_metadata = {
                "backend": "numpy",
                "method": "mergeable_streaming_moments_with_full_stream_log_quantiles",
                "approximate_quantiles": True,
                "quantile_relative_accuracy": 0.01,
                "quantile_source": "full_stream_sketch",
                "quantile_cell_budget": int(PYTHON_NUMERIC_SKETCH_CELL_BUDGET),
                "finite_only_moments": True,
                "columns_profiled": len(numeric_cols),
                "raw_observations_retained": False,
            }
        else:
            numeric_metadata = {
                "backend": "numpy",
                "method": "mergeable_streaming_moments_with_row_sample_quantiles",
                "approximate_quantiles": len(sample) < rows,
                "quantile_sample_rows": int(len(sample)),
                "quantile_source": "bounded_row_sample",
                "quantile_cell_budget": int(PYTHON_NUMERIC_SKETCH_CELL_BUDGET),
                "quantile_sketch_skipped_for_cost": bool(numeric_cols),
                "finite_only_moments": True,
                "columns_profiled": len(numeric_cols),
                "raw_observations_retained": False,
            }

    categorical_summary: dict[str, dict[str, Any]] = {}
    native_categorical_columns: list[str] = []
    sample_categorical_columns: list[str] = []
    for column in categorical_cols:
        string_accumulator = native_string_accumulators.get(column)
        if string_accumulator is not None:
            categorical_payload = dict(string_accumulator.snapshot())
            missing_counts[column] = int(categorical_payload["missing"])
            count = int(categorical_payload["count"])
            estimate = min(int(categorical_payload["cardinality_estimate"]), count)
            categorical_summary[column] = {
                "unique_values": estimate,
                "top_values": {
                    str(label): int(candidate_count)
                    for label, candidate_count in categorical_payload["heavy_hitters"][:10]
                },
                "approximate": True,
                "unique_values_method": categorical_payload["cardinality_method"],
                "top_values_method": categorical_payload["heavy_hitter_method"],
                "top_values_count_semantics": categorical_payload[
                    "heavy_hitter_count_semantics"
                ],
            }
            native_categorical_columns.append(column)
            continue

        if column not in sample:
            continue
        counts = sample[column].value_counts(dropna=False).head(10)
        categorical_summary[column] = {
            "unique_values": int(sample[column].nunique(dropna=True)),
            "top_values": {str(key): int(value) for key, value in counts.items()},
            "approximate": len(sample) < rows,
        }
        sample_categorical_columns.append(column)

    if native_categorical_columns:
        categorical_method = (
            "native_full_stream_sketch"
            if not sample_categorical_columns
            else "native_full_stream_sketch_with_sample_fallback"
        )
    else:
        categorical_method = "exact" if len(sample) == rows else "evenly_spaced_row_sample"
    categorical_metadata = {
        "method": categorical_method,
        "sampled": bool(sample_categorical_columns and len(sample) < rows),
        "sample_rows": int(len(sample)),
        "source_rows": int(rows),
        "native_full_stream_columns": native_categorical_columns,
        "sample_fallback_columns": sample_categorical_columns,
        "cardinality_method": "hyperloglog" if native_categorical_columns else None,
        "top_values_count_semantics": (
            "lower_bound_candidates" if native_categorical_columns else None
        ),
    }

    missing_series = pd.Series(missing_counts, dtype="int64")
    missing_percent = (missing_series / max(rows, 1) * 100).round(2)

    duplicate_rows, duplicate_metadata = _duplicate_summary(sample, int(rows))
    correlations, correlation_metadata = _bounded_correlations(sample, numeric_cols)
    correlation_metadata.update({
        "row_sampled": len(sample) < rows,
        "sample_rows": int(len(sample)),
        "source_rows": int(rows),
    })

    memory_usage_mb, memory_usage_metadata = _estimated_memory_usage(sample, int(rows))
    preview_source = preview_frame if preview_frame is not None else pd.DataFrame(columns=columns)
    preview_object = preview_source.astype(object)
    preview = preview_object.where(preview_object.notna(), None).to_dict(orient="records")

    source_metadata = metadata.to_dict() if hasattr(metadata, "to_dict") else asdict(metadata)
    payload = {
        "shape": {"rows": int(rows), "columns": len(columns)},
        "columns": columns,
        "dtypes": dtypes,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "missing_counts": series_to_dict(missing_series),
        "missing_percent": series_to_dict(missing_percent),
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": round(duplicate_rows / max(rows, 1) * 100, 2),
        "duplicate_metadata": duplicate_metadata,
        "memory_usage_mb": memory_usage_mb,
        "memory_usage_metadata": memory_usage_metadata,
        "numeric_summary": numeric_summary,
        "numeric_summary_metadata": numeric_metadata,
        "categorical_summary": categorical_summary,
        "categorical_summary_metadata": categorical_metadata,
        "correlations": correlations,
        "correlation_metadata": correlation_metadata,
        "preview": preview,
        "source_metadata": source_metadata,
        "streaming_metadata": {
            "enabled": True,
            "full_materialization": False,
            "batch_size": int(effective_batch_size),
            "requested_batch_size": int(batch_size),
            "batch_cell_budget": int(STREAM_BATCH_CELL_BUDGET),
            "batches_scanned": int(batches_scanned),
            "sample_rows": int(len(sample)),
            "sample_row_limit": int(effective_sample_rows),
            "requested_sample_rows": int(sample_rows),
            "sample_cell_budget": int(STREAM_SAMPLE_CELL_BUDGET),
            "sample_cells_retained": int(len(sample) * len(columns)),
            "source_columns": int(len(columns)),
            "width_limited": bool(
                effective_batch_size < int(batch_size)
                or effective_sample_rows < int(sample_rows)
            ),
            "sample_strategy": "evenly_spaced_global_rows",
            "numeric_backend": selected_backend,
            "numpy_full_stream_quantile_sketches": bool(use_numpy_quantile_sketches),
            "numpy_numeric_sketch_cell_budget": int(PYTHON_NUMERIC_SKETCH_CELL_BUDGET),
            "native_string_sketch_columns": native_categorical_columns,
        },
    }
    if return_sample:
        return payload, sample
    return payload
