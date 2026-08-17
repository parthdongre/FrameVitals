import pandas as pd
import pytest

from framevitals.ml_readiness import calculate_ml_readiness_from_profile
from framevitals.streaming_quality import run_streaming_quality_diagnostics


def _projected_profile(*, rows: int = 100, source_columns: int = 10_000, profiled_columns: int = 64):
    columns = [f"c{i}" for i in range(profiled_columns)]
    return {
        "shape": {"rows": rows, "columns": source_columns},
        "columns": columns,
        "numeric_columns": columns,
        "categorical_columns": [],
        "missing_counts": {name: rows for name in columns},
        "missing_percent": {name: 100.0 for name in columns},
        "duplicate_rows": 0,
        "duplicate_percent": 0.0,
        "duplicate_metadata": {"sampled": True},
        "streaming_metadata": {
            "enabled": True,
            "full_materialization": False,
            "sample_rows": min(rows, 50),
            "source_columns": source_columns,
            "profiled_columns": profiled_columns,
            "column_sampled": True,
        },
    }


def test_ml_readiness_uses_profiled_width_for_projected_missingness():
    profile = _projected_profile()
    result = calculate_ml_readiness_from_profile(profile)

    assert result["issues"]["missing_percent"] == pytest.approx(100.0)
    assert result["score_scope"] == "full_rows_projected_columns_estimate"
    assert result["profiled_columns"] == 64
    assert result["source_columns"] == 10_000
    assert result["execution"]["components"]["missingness"] == (
        "full_rows_projected_columns_exact"
    )
    assert result["execution"]["components"]["column_groups"] == "projected_schema_exact"
    assert result["execution"]["components"]["duplicate_rate"] == (
        "bounded_row_sample_projected_columns_estimate"
    )


def test_streaming_quality_reports_only_columns_actually_checked():
    profile = _projected_profile(rows=200, profiled_columns=64)
    sample = pd.DataFrame({f"c{i}": [i, i + 1, i + 2] for i in range(64)})

    result = run_streaming_quality_diagnostics(
        sample,
        profile=profile,
        source_rows=200,
        source_columns=10_000,
        max_sample_rows=3,
        max_columns=100,
    )

    assert result["columns"] == 10_000
    assert result["profiled_columns"] == 64
    assert result["columns_checked"] == 64
    assert result["truncated_columns"] is True
    execution = result["execution"]
    assert execution["column_sampled"] is True
    assert execution["profile_fact_scope"] == "full_rows_projected_columns"
    assert execution["full_source_inputs"] == []
    assert set(execution["projected_full_row_inputs"]) == {
        "missingness",
        "duplicate_row_estimate",
    }
