import pandas as pd

import framevitals
from framevitals.config import resolve_config


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "x": list(range(rows)),
        "y": [index * 2 for index in range(rows)],
        "group": ["a", "b"] * (rows // 2),
    })


def test_explicit_resource_caps_override_environment_and_config(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_MAX_SAMPLE_ROWS", "90")
    monkeypatch.setenv("FRAMEVITALS_MAX_RELATIONSHIP_PAIRS", "80")

    resolved = resolve_config(
        {
            "resources": {
                "max_sample_rows": 70,
                "max_relationship_pairs": 60,
                "max_memory_heavy_parallelism": 4,
                "max_streaming_profile_columns": 50,
            }
        },
        max_sample_rows=7,
        max_relationship_pairs=6,
        max_memory_heavy_parallelism=1,
        max_streaming_profile_columns=5,
    )

    assert resolved.max_sample_rows == 7
    assert resolved.max_relationship_pairs == 6
    assert resolved.max_memory_heavy_parallelism == 1
    assert resolved.max_streaming_profile_columns == 5


def test_public_plan_accepts_explicit_resource_caps():
    plan = framevitals.plan(
        _frame(),
        mode="standard",
        max_sample_rows=5,
        max_relationship_pairs=3,
        max_memory_heavy_parallelism=1,
        max_streaming_profile_columns=2,
    )

    assert plan.resource_policy == {
        "max_sample_rows": 5,
        "max_relationship_pairs": 3,
        "max_memory_heavy_parallelism": 1,
        "max_streaming_profile_columns": 2,
    }
    budget = plan.execution_budget
    assert budget["quality_sample_rows"] <= 5
    assert budget["deep_statistics_sample_rows"] <= 5
    assert budget["anomaly_sample_rows"] <= 5
    assert budget["time_series_sample_rows"] <= 5
    assert budget["relationship_pair_budget"] <= 3
    assert budget["max_memory_heavy_parallelism"] == 1


def test_public_analyze_accepts_explicit_resource_caps():
    result = framevitals.analyze(
        _frame(),
        mode="standard",
        max_sample_rows=6,
        max_relationship_pairs=2,
        max_memory_heavy_parallelism=1,
    )

    assert result["execution"]["resource_policy"]["max_sample_rows"] == 6
    assert result["execution"]["resource_policy"]["max_relationship_pairs"] == 2
    budget = result["execution"]["budget"]
    assert budget["quality_sample_rows"] <= 6
    assert budget["anomaly_sample_rows"] <= 6
    assert budget["relationship_pair_budget"] <= 2
    assert budget["max_memory_heavy_parallelism"] == 1
