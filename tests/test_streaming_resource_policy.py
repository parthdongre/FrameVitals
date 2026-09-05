import pandas as pd

from framevitals.execution import ExecutionPolicy, use_execution_policy
from framevitals.streaming_quality import run_streaming_quality_diagnostics


def test_streaming_quality_never_widens_hard_sample_cap(monkeypatch):
    captured = {}

    def fake_quality(*args, **kwargs):
        captured["max_sample_rows"] = kwargs["max_sample_rows"]
        return {
            "identifier_duplicates": [],
            "quasi_constant_columns": [],
            "duplicate_columns": [],
            "coercion_candidates": [],
            "category_normalisation": [],
            "blank_strings": [],
            "infinite_values": [],
            "mixed_object_types": [],
            "missingness_relationships": [],
            "primary_key_candidates": [],
        }

    monkeypatch.setattr(
        "framevitals.streaming_quality.run_quality_diagnostics",
        fake_quality,
    )

    sample = pd.DataFrame({"x": range(20)})
    profile = {
        "columns": ["x"],
        "missing_counts": {"x": 0},
        "duplicate_rows": 0,
    }

    with use_execution_policy(ExecutionPolicy(max_sample_rows=4)):
        run_streaming_quality_diagnostics(
            sample,
            profile=profile,
            source_rows=100,
            source_columns=1,
            max_sample_rows=10,
        )

    assert captured["max_sample_rows"] == 4
