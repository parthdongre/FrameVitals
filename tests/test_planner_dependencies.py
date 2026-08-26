import pandas as pd

import framevitals
from framevitals.planner import plan_execution_modules


def _signals() -> dict[str, bool]:
    return {
        "has_numeric_columns": True,
        "has_datetime_columns": False,
        "has_time_series_structure": False,
        "has_long_text_columns": False,
    }


def test_disabled_dependency_blocks_downstream_modules():
    modules = plan_execution_modules(
        signals=_signals(),
        analysis_mode="research",
        target_column="target",
        disabled_modules=["target_intelligence"],
    )
    decisions = modules["decisions"]

    assert decisions["target_intelligence"]["status"] == "disabled_by_config"
    assert decisions["modeling"]["status"] == "not_applicable"
    assert decisions["modeling"]["blocked_by"] == ["target_intelligence"]
    assert "target_intelligence (disabled_by_config)" in decisions["modeling"]["reason"]
    assert decisions["explainability"]["status"] == "not_applicable"
    assert decisions["explainability"]["blocked_by"] == ["modeling"]


def test_execution_stages_respect_dependency_order():
    modules = plan_execution_modules(
        signals=_signals(),
        analysis_mode="research",
        target_column="target",
    )
    stages = modules["execution_stages"]
    positions = {
        module: stage["stage"]
        for stage in stages
        for module in stage["modules"]
    }

    assert positions["target_intelligence"] < positions["modeling"]
    assert positions["modeling"] < positions["explainability"]
    assert modules["runnable_modules"] == [
        module
        for stage in stages
        for module in stage["modules"]
    ]


def test_plan_exposes_scheduler_ready_stages():
    frame = pd.DataFrame({
        "x": list(range(30)),
        "y": [index * 2 for index in range(30)],
        "target": [0, 1] * 15,
    })

    plan = framevitals.plan(frame, mode="research", target="target")
    modules = plan.execution_modules
    stages = modules["execution_stages"]

    assert stages
    assert all("resource_classes" in stage for stage in stages)
    assert "target_intelligence" in modules["runnable_modules"]
    assert "modeling" in modules["runnable_modules"]
    assert "explainability" in modules["runnable_modules"]
