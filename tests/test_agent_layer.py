import pandas as pd

from framevitals.agent_brief import (
    build_dataset_brief,
    render_brief_block,
)
from framevitals.agent_tools import (
    AGENT_TOOLS,
    AgentContext,
    list_tools,
    run_tool,
)
from framevitals.rag_index import (
    build_fact_index,
)


def make_result():
    return {
        "id": "demo-1",
        "filename": "customers.csv",
        "analysisMode": "standard",
        "rows": 4,
        "columns": 3,
        "profile": {
            "shape": {
                "rows": 4,
                "columns": 3,
            },
            "dtypes": {
                "age": "int64",
                "region": "object",
                "revenue": "int64",
            },
            "numeric_columns": [
                "age",
                "revenue",
            ],
            "categorical_columns": [
                "region",
            ],
            "date_columns": [],
            "missing_percent": {
                "age": 0.0,
                "region": 0.0,
                "revenue": 0.0,
            },
            "duplicate_rows": 0,
            "categorical_summary": {
                "region": {
                    "unique_count": 2,
                    "top_values": {
                        "North": 2,
                        "South": 2,
                    },
                },
            },
            "preview": [
                {
                    "age": 20,
                    "region": "North",
                    "revenue": 100,
                },
                {
                    "age": 30,
                    "region": "South",
                    "revenue": 200,
                },
            ],
        },
        "columnRoles": {
            "age": {
                "roles": ["numeric"],
            },
            "region": {
                "roles": ["categorical"],
            },
            "revenue": {
                "roles": ["numeric"],
            },
        },
        "health": {
            "overall_score": 92,
            "label": "Excellent",
            "components": {
                "completeness": 100,
                "consistency": 95,
            },
        },
        "mlReadiness": {
            "score": 88,
            "label": "Ready",
            "recommendations": [],
        },
        "signals": [
            {
                "name": "healthy_dataset",
                "severity": "low",
                "evidence": (
                    "No major quality issues."
                ),
                "recommendation": (
                    "Proceed with analysis."
                ),
            },
        ],
    }


def make_df():
    return pd.DataFrame({
        "age": [
            20,
            30,
            40,
            50,
        ],
        "region": [
            "North",
            "South",
            "North",
            "South",
        ],
        "revenue": [
            100,
            200,
            300,
            400,
        ],
    })


def test_build_dataset_brief():
    brief = build_dataset_brief(
        make_result()
    )

    assert brief["dataset"]["id"] == (
        "demo-1"
    )

    assert brief["dataset"]["filename"] == (
        "customers.csv"
    )

    assert brief["quality"][
        "health_score"
    ] == 92.0

    assert brief["schema"][
        "numeric_columns"
    ] == [
        "age",
        "revenue",
    ]

    assert brief["sample_rows"]


def test_render_brief_block():
    brief = build_dataset_brief(
        make_result()
    )

    block = render_brief_block(
        brief
    )

    assert "Dataset brief" in block
    assert "customers.csv" in block
    assert "```json" in block


def test_agent_tool_registry():
    expected = {
        "get_section",
        "list_columns",
        "column_summary",
        "run_query",
        "get_top_anomalies",
        "get_leaderboard",
        "get_explainability_top",
        "search_facts",
        "get_dataset_brief",
    }

    assert set(
        AGENT_TOOLS
    ) == expected

    descriptors = list_tools()

    assert {
        item["name"]
        for item in descriptors
    } == expected


def test_agent_list_columns_tool():
    result = make_result()

    ctx = AgentContext(
        df=make_df(),
        analysis_result=result,
    )

    output = run_tool(
        "list_columns",
        {},
        ctx,
    )

    assert output["ok"] is True

    names = [
        item["name"]
        for item in output["columns"]
    ]

    assert names == [
        "age",
        "region",
        "revenue",
    ]


def test_agent_safe_query_tool():
    ctx = AgentContext(
        df=make_df(),
        analysis_result=make_result(),
    )

    output = run_tool(
        "run_query",
        {
            "expression": (
                "df['revenue'].mean()"
            ),
        },
        ctx,
    )

    assert output["ok"] is True
    assert output["result"] == 250.0


def test_agent_get_section_tool():
    ctx = AgentContext(
        df=make_df(),
        analysis_result=make_result(),
    )

    output = run_tool(
        "get_section",
        {
            "path": (
                "health.overall_score"
            ),
        },
        ctx,
    )

    assert output == {
        "ok": True,
        "path": (
            "health.overall_score"
        ),
        "value": 92,
    }


def test_agent_dataset_brief_tool():
    ctx = AgentContext(
        df=make_df(),
        analysis_result=make_result(),
    )

    output = run_tool(
        "get_dataset_brief",
        {},
        ctx,
    )

    assert output["ok"] is True

    assert output["brief"][
        "dataset"
    ]["filename"] == (
        "customers.csv"
    )


def test_agent_search_facts_tool(
    monkeypatch,
):
    monkeypatch.setenv(
        "DATALENS_RAG_BACKEND",
        "tfidf",
    )

    result = make_result()

    facts = build_fact_index(
        result
    )

    ctx = AgentContext(
        df=make_df(),
        analysis_result=result,
        facts=facts,
    )

    output = run_tool(
        "search_facts",
        {
            "question": (
                "What is the health score?"
            ),
            "k": 3,
        },
        ctx,
    )

    assert output["ok"] is True
    assert output["backend"] == "tfidf"
    assert output["facts"]
    assert output["rendered"]


def test_unknown_agent_tool():
    ctx = AgentContext(
        df=make_df(),
        analysis_result=make_result(),
    )

    output = run_tool(
        "does_not_exist",
        {},
        ctx,
    )

    assert output["ok"] is False
