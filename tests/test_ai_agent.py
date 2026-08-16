import pandas as pd
import pytest

pytest.importorskip("pydantic")

import framevitals.ai_agent as ai_agent
from framevitals.ai_agent import answer_with_agent


def make_result():
    return {
        "id": "agent-demo",
        "filename": "customers.csv",
        "rows": 4,
        "columns": 2,
        "profile": {
            "shape": {
                "rows": 4,
                "columns": 2,
            },
            "columns": [
                "age",
                "revenue",
            ],
            "dtypes": {
                "age": "int64",
                "revenue": "int64",
            },
            "numeric_columns": [
                "age",
                "revenue",
            ],
            "categorical_columns": [],
            "date_columns": [],
            "missing_percent": {
                "age": 0.0,
                "revenue": 0.0,
            },
            "duplicate_rows": 0,
            "preview": [
                {
                    "age": 20,
                    "revenue": 100,
                },
                {
                    "age": 30,
                    "revenue": 200,
                },
            ],
        },
        "health": {
            "overall_score": 94,
            "label": "Excellent",
        },
        "mlReadiness": {
            "score": 90,
            "label": "Ready",
        },
    }


def make_df():
    return pd.DataFrame({
        "age": [
            20,
            30,
            40,
            50,
        ],
        "revenue": [
            100,
            200,
            300,
            400,
        ],
    })


def test_empty_question_uses_fallback():
    result = answer_with_agent(
        question="",
        df=make_df(),
        analysis_result=make_result(),
    )

    assert result["source"] == (
        "fallback"
    )

    assert result["answer"] == (
        "Please provide a question."
    )

    assert result["trace"] == {}


def test_fast_agent_falls_back_without_llm(
    monkeypatch,
):
    monkeypatch.setenv(
        "DATALENS_RAG_BACKEND",
        "tfidf",
    )

    def fail_llm(
        messages,
        json_mode=False,
    ):
        raise RuntimeError(
            "LLM disabled for test"
        )

    monkeypatch.setattr(
        ai_agent,
        "_call_llm",
        fail_llm,
    )

    result = answer_with_agent(
        question=(
            "What is the dataset "
            "health score?"
        ),
        df=make_df(),
        analysis_result=make_result(),
        fast=True,
    )

    assert result["source"] == (
        "fallback"
    )

    assert result["answer"]

    assert result["trace"]["fast"] is (
        True
    )

    assert result["trace"][
        "rag_backend"
    ] == "tfidf"

    assert result["trace"][
        "tool_calls"
    ]
