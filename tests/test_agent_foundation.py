import pandas as pd

from framevitals.rag_index import (
    build_fact_index,
    render_facts_block,
    retrieve,
)
from framevitals.safe_pandas import (
    safe_eval,
)


def make_dataframe():
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


def test_safe_eval_numeric_expression():
    result = safe_eval(
        "df['age'].mean()",
        make_dataframe(),
    )

    assert result["ok"] is True
    assert result["error"] is None
    assert result["result"] == 35.0


def test_safe_eval_groupby():
    result = safe_eval(
        (
            "df.groupby('region')"
            "['revenue'].sum()"
        ),
        make_dataframe(),
    )

    assert result["ok"] is True

    series = result["result"]

    assert series["type"] == "series"
    assert series["length"] == 2

    values = {
        row["index"]: row["value"]
        for row in series["rows"]
    }

    assert values == {
        "North": 400,
        "South": 600,
    }


def test_safe_eval_rejects_dangerous_code():
    result = safe_eval(
        "__import__('os').system('echo bad')",
        make_dataframe(),
    )

    assert result["ok"] is False
    assert result["result"] is None
    assert result["error"]


def test_safe_eval_dataframe_result():
    result = safe_eval(
        "df.head(2)",
        make_dataframe(),
    )

    assert result["ok"] is True

    output = result["result"]

    assert output["type"] == "dataframe"
    assert output["shape"] == [2, 3]
    assert len(output["rows"]) == 2


def make_analysis_result():
    return {
        "profile": {
            "rows": 100,
            "columns": [
                "age",
                "income",
            ],
            "preview": [
                {
                    "age": 20,
                    "income": 50000,
                },
            ],
        },
        "health": {
            "overall_score": 88,
            "label": "Good",
        },
        "ml_readiness": {
            "score": 81,
            "label": "Ready",
        },
        "signals": [
            {
                "severity": "high",
                "title": "Missing values",
            },
        ],
        "charts": [
            {
                "path": "chart.png",
            },
        ],
    }


def test_build_fact_index_excludes_noisy_paths():
    facts = build_fact_index(
        make_analysis_result()
    )

    paths = {
        fact.path
        for fact in facts
    }

    assert "health.overall_score" in paths
    assert "health.label" in paths

    assert not any(
        path.startswith("profile.preview")
        for path in paths
    )

    assert not any(
        path.startswith("charts")
        for path in paths
    )


def test_tfidf_fact_retrieval(monkeypatch):
    monkeypatch.setenv(
        "DATALENS_RAG_BACKEND",
        "tfidf",
    )

    facts = build_fact_index(
        make_analysis_result()
    )

    result = retrieve(
        "What is the dataset health score?",
        facts,
        k=3,
    )

    assert result["backend"] == "tfidf"
    assert result["k"] > 0

    paths = {
        fact["path"]
        for fact in result["facts"]
    }

    assert "health.overall_score" in paths


def test_render_facts_block(monkeypatch):
    monkeypatch.setenv(
        "DATALENS_RAG_BACKEND",
        "tfidf",
    )

    facts = build_fact_index(
        make_analysis_result()
    )

    retrieved = retrieve(
        "health score",
        facts,
        k=2,
    )

    block = render_facts_block(
        retrieved
    )

    assert block
    assert block != "(no relevant facts)"
    assert "health" in block.lower()
