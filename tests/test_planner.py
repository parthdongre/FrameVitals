import pandas as pd

import framevitals
from framevitals.planner import (
    PLANNER_SCHEMA_VERSION,
    effective_disabled_modules,
    plan_execution_modules,
)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "age": list(range(20, 40)),
        "income": [30_000 + index * 1_000 for index in range(20)],
        "city": ["Pune", "Mumbai"] * 10,
        "churn": [0, 1] * 10,
    })


def test_module_planner_reports_versioned_explainable_decisions():
    result = framevitals.plan(_dataset(), mode="standard")

    modules = result.selection["execution_modules"]
    decisions = modules["decisions"]

    assert result.planner_schema_version == PLANNER_SCHEMA_VERSION
    assert result.summary()["planner_schema_version"] == PLANNER_SCHEMA_VERSION
    assert decisions["quality_diagnostics"]["status"] == "run"
    assert decisions["anomaly_detection"]["status"] == "run"
    assert decisions["deep_statistics"]["status"] == "disabled_by_mode"
    assert decisions["text_profile"]["status"] == "disabled_by_mode"
    assert decisions["target_intelligence"]["status"] == "not_applicable"
    assert decisions["charts"]["status"] == "not_applicable"
    assert decisions["ai"]["status"] == "conditional"
    assert "Execution modules" in result.explain_text()


def test_explicit_disable_is_distinct_from_mode_policy():
    result = framevitals.plan(
        _dataset(),
        mode="standard",
        disabled_modules=["anomaly_detection"],
    )
    modules = result.execution_modules

    assert modules["decisions"]["anomaly_detection"]["status"] == "disabled_by_config"
    assert "anomaly_detection" in modules["explicit_disabled"]
    assert "anomaly_detection" in modules["effective_disabled"]
    assert modules["decisions"]["deep_statistics"]["status"] == "disabled_by_mode"
    assert "deep_statistics" not in modules["explicit_disabled"]
    assert "deep_statistics" in modules["effective_disabled"]


def test_target_and_research_mode_unlock_dependent_modules():
    result = framevitals.plan(_dataset(), mode="research", target="churn")
    decisions = result.module_decisions

    assert decisions["target_intelligence"]["status"] == "run"
    assert decisions["modeling"]["status"] == "run"
    assert decisions["modeling"]["depends_on"] == ["target_intelligence"]
    assert decisions["explainability"]["status"] == "conditional"
    assert decisions["explainability"]["depends_on"] == ["modeling"]


def test_mode_policy_helper_is_the_runtime_source_of_truth():
    disabled = effective_disabled_modules("quick", ("charts",))

    assert "charts" in disabled
    assert "deep_statistics" in disabled
    assert "anomaly_detection" in disabled
    assert "target_intelligence" not in disabled


def test_signal_applicability_reasons_are_structured():
    modules = plan_execution_modules(
        signals={
            "has_numeric_columns": False,
            "has_datetime_columns": False,
            "has_time_series_structure": False,
            "has_long_text_columns": False,
        },
        analysis_mode="deep",
        target_column=None,
    )

    assert modules["decisions"]["anomaly_detection"]["status"] == "not_applicable"
    assert "has_numeric_columns" in modules["decisions"]["anomaly_detection"]["reason"]
    assert modules["decisions"]["time_series"]["status"] == "not_applicable"
