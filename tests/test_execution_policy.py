import pandas as pd
import pytest

import framevitals
from framevitals.config import available_presets, resolve_config
from framevitals.execution import (
    ExecutionPolicy,
    derive_execution_budget,
    derive_streaming_profile_column_limit,
    use_execution_policy,
)


def _small_dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "x": list(range(20)),
        "y": [value * 2 for value in range(20)],
        "group": ["a", "b"] * 10,
    })


def test_resource_caps_resolve_from_config_and_exhaustive_alias():
    resolved = resolve_config(
        preset="exhaustive",
        config={
            "resources": {
                "max_sample_rows": 1200,
                "max_relationship_pairs": 7,
                "max_memory_heavy_parallelism": 1,
                "max_streaming_profile_columns": 12,
            }
        },
    )

    assert "exhaustive" in available_presets()
    assert resolved.mode == "research"
    assert resolved.max_sample_rows == 1200
    assert resolved.max_relationship_pairs == 7
    assert resolved.max_memory_heavy_parallelism == 1
    assert resolved.max_streaming_profile_columns == 12
    assert resolved.to_dict()["max_sample_rows"] == 1200

    with pytest.raises(ValueError, match="max_sample_rows"):
        resolve_config({"resources": {"max_sample_rows": 0}})


def test_execution_policy_caps_budgets_and_restores_context():
    baseline = derive_execution_budget(100_000, 200, mode="research")
    policy = ExecutionPolicy(
        max_sample_rows=1200,
        max_relationship_pairs=7,
        max_memory_heavy_parallelism=1,
        max_streaming_profile_columns=12,
    )

    with use_execution_policy(policy):
        capped = derive_execution_budget(100_000, 200, mode="research")
        profile_columns = derive_streaming_profile_column_limit(
            10_000_000,
            10_000,
            mode="research",
        )

    assert capped.quality_sample_rows <= 1200
    assert capped.deep_statistics_sample_rows <= 1200
    assert capped.bootstrap_sample_rows <= 1200
    assert capped.distribution_sample_rows <= 1200
    assert capped.pair_sample_rows <= 1200
    assert capped.anomaly_sample_rows <= 1200
    assert capped.time_series_sample_rows <= 1200
    assert capped.relationship_pair_budget == 7
    assert capped.max_memory_heavy_parallelism == 1
    assert profile_columns <= 12
    assert derive_execution_budget(100_000, 200, mode="research") == baseline


def test_policy_never_expands_mode_defaults():
    baseline = derive_execution_budget(100_000, 20, mode="quick")
    policy = ExecutionPolicy(
        max_sample_rows=999_999,
        max_relationship_pairs=999_999,
        max_memory_heavy_parallelism=999,
    )

    with use_execution_policy(policy):
        capped = derive_execution_budget(100_000, 20, mode="quick")

    assert capped.quality_sample_rows == baseline.quality_sample_rows
    assert capped.anomaly_sample_rows == baseline.anomaly_sample_rows
    assert capped.relationship_pair_budget == baseline.relationship_pair_budget
    assert (
        capped.max_memory_heavy_parallelism
        == baseline.max_memory_heavy_parallelism
    )


def test_plan_reports_effective_resource_budget():
    result = framevitals.plan(
        _small_dataset(),
        mode="research",
        config={
            "resources": {
                "max_sample_rows": 8,
                "max_relationship_pairs": 3,
                "max_memory_heavy_parallelism": 1,
            }
        },
    )

    assert result.resource_policy["max_sample_rows"] == 8
    assert result.execution_budget["quality_sample_rows"] == 8
    assert result.execution_budget["deep_statistics_sample_rows"] == 8
    assert result.execution_budget["relationship_pair_budget"] == 3
    assert result.execution_budget["max_memory_heavy_parallelism"] == 1
    assert "Resource caps" in result.explain_text()


def test_analyze_applies_policy_to_pipeline_budget(monkeypatch):
    captured = {}

    def fake_run_full_analysis(**kwargs):
        budget = derive_execution_budget(100_000, 64, mode=kwargs["analysis_mode"])
        captured["budget"] = budget
        return {
            "profile": {"shape": {"rows": 1, "columns": 1}},
            "execution": {},
            "signals": [],
        }

    monkeypatch.setattr(
        "framevitals.analysis_api.run_full_analysis",
        fake_run_full_analysis,
    )

    result = framevitals.analyze(
        pd.DataFrame({"x": [1]}),
        mode="research",
        config={
            "resources": {
                "max_sample_rows": 17,
                "max_relationship_pairs": 2,
                "max_memory_heavy_parallelism": 1,
            }
        },
    )

    budget = captured["budget"]
    assert budget.quality_sample_rows == 17
    assert budget.anomaly_sample_rows == 17
    assert budget.relationship_pair_budget == 2
    assert budget.max_memory_heavy_parallelism == 1
    assert result["execution"]["resource_policy"]["max_sample_rows"] == 17


def test_environment_precedence_is_deterministic(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_PRESET", "deep")
    monkeypatch.setenv("FRAMEVITALS_MODE", "research")
    monkeypatch.setenv("FRAMEVITALS_WORKERS", "2")
    monkeypatch.setenv("FRAMEVITALS_ARTIFACTS", "true")
    monkeypatch.setenv("FRAMEVITALS_MAX_SAMPLE_ROWS", "900")
    monkeypatch.setenv(
        "FRAMEVITALS_DISABLED_MODULES",
        "modeling, explainability",
    )

    from_environment = resolve_config(preset="quick")
    assert from_environment.mode == "research"
    assert from_environment.workers == 2
    assert from_environment.artifacts is True
    assert from_environment.max_sample_rows == 900
    assert from_environment.disabled_modules == ("modeling", "explainability")

    configured = resolve_config(
        config={
            "analysis": {"mode": "standard", "artifacts": False},
            "resources": {"workers": 3, "max_sample_rows": 400},
        },
        mode="deep",
    )
    assert configured.mode == "deep"
    assert configured.workers == 3
    assert configured.artifacts is False
    assert configured.max_sample_rows == 400

    monkeypatch.setenv("FRAMEVITALS_ARTIFACTS", "sometimes")
    with pytest.raises(ValueError, match="FRAMEVITALS_ARTIFACTS"):
        resolve_config()
