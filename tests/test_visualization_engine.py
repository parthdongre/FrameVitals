import pandas as pd

import framevitals.visualizer as visualizer_module

from framevitals.chart_planner import (
    build_chart_plan,
)
from framevitals.column_roles import (
    infer_column_roles,
)
from framevitals.profiler import (
    build_profile,
)


def make_dataset():
    return pd.DataFrame({
        "age": [
            20, 21, 22, 23, 24,
        ],
        "income": [
            30000,
            32000,
            35000,
            37000,
            40000,
        ],
        "city": [
            "Pune",
            "Mumbai",
            "Pune",
            "Nashik",
            "Mumbai",
        ],
    })


def test_chart_planner():
    df = make_dataset()

    profile = build_profile(df)
    roles = infer_column_roles(df)

    plan = build_chart_plan(
        df=df,
        profile=profile,
        health={},
        advanced={},
        cleaning={},
        column_roles=roles,
    )

    types = {
        item["type"]
        for item in plan
    }

    assert "health_components" in types
    assert "dtype_breakdown" in types
    assert "cardinality_strip" in types
    assert "numeric_distribution" in types


def test_visualizer_creates_chart_directory(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    df = make_dataset()

    assert not (
        tmp_path
        / "static"
        / "charts"
    ).exists()

    monkeypatch.setattr(
        visualizer_module,
        "build_chart_plan",
        lambda *args, **kwargs: [
            {
                "id": "health",
                "type": "health_components",
                "title": "Dataset Health Components",
                "reason": "test chart",
            },
        ],
    )

    charts = visualizer_module.generate_charts(
        dataset_id="demo",
        df=df,
        health={
            "components": {
                "completeness": 100,
                "consistency": 90,
                "uniqueness": 95,
                "outlier_safety": 88,
            },
        },
        advanced={},
        cleaning={},
    )

    assert len(charts) == 1

    assert charts[0]["type"] == (
        "health_components"
    )

    output = (
        tmp_path
        / "static"
        / "charts"
        / "demo_health_components.png"
    )

    assert output.exists()
