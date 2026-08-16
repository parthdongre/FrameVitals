from __future__ import annotations

import pandas as pd
import pytest

import framevitals.analysis_api as analysis_api
from framevitals.analysis_selector import select_analyses


@pytest.mark.parametrize(
    ("mode", "expected_mode_disabled"),
    [
        (
            "quick",
            {
                "deep_statistics",
                "anomaly_detection",
                "time_series",
                "text_profile",
                "modeling",
                "explainability",
            },
        ),
        (
            "standard",
            {"deep_statistics", "text_profile", "modeling", "explainability"},
        ),
        ("deep", {"modeling", "explainability"}),
        ("research", set()),
    ],
)
def test_mode_policy_is_explicit_and_stable(mode, expected_mode_disabled):
    assert set(analysis_api._MODE_DISABLED_MODULES[mode]) == expected_mode_disabled


def test_effective_mode_policy_preserves_explicit_user_disables():
    disabled = analysis_api._effective_disabled_modules(
        "standard",
        ("time_series", "charts"),
    )
    assert set(disabled) == {
        "deep_statistics",
        "text_profile",
        "modeling",
        "explainability",
        "time_series",
        "charts",
    }


def test_quick_keeps_target_intelligence_and_cleaning_available(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_full_analysis(**kwargs):
        captured.update(kwargs)
        return {
            "filename": "<dataframe>",
            "profile": {"shape": {"rows": 3, "columns": 2}},
            "execution": {},
        }

    monkeypatch.setattr(analysis_api, "run_full_analysis", fake_run_full_analysis)
    result = analysis_api.analyze(
        pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]}),
        mode="quick",
        target="target",
        artifacts=True,
    )

    disabled = set(captured["disabled_modules"])
    assert "target_intelligence" not in disabled
    assert "cleaning" not in disabled
    assert {
        "deep_statistics",
        "anomaly_detection",
        "time_series",
        "text_profile",
        "modeling",
        "explainability",
    } <= disabled
    assert result["config"] == {
        "mode": "quick",
        "target": "target",
        "artifacts": True,
        "workers": result["config"]["workers"],
        "disabled_modules": (),
    }


def test_standard_public_analysis_does_not_schedule_deep_only_modules(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_full_analysis(**kwargs):
        captured.update(kwargs)
        return {
            "filename": "<dataframe>",
            "profile": {"shape": {"rows": 3, "columns": 2}},
            "execution": {},
        }

    monkeypatch.setattr(analysis_api, "run_full_analysis", fake_run_full_analysis)
    result = analysis_api.analyze(
        pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]}),
        mode="standard",
        target="target",
        artifacts=False,
    )

    disabled = set(captured["disabled_modules"])
    assert {"deep_statistics", "text_profile", "modeling", "explainability"} <= disabled
    assert "anomaly_detection" not in disabled
    assert "time_series" not in disabled
    assert "target_intelligence" not in disabled
    assert result["config"]["mode"] == "standard"
    assert result["config"]["disabled_modules"] == ()


def test_deep_keeps_advanced_diagnostics_but_reserves_modeling_for_research(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_full_analysis(**kwargs):
        captured.update(kwargs)
        return {
            "filename": "<dataframe>",
            "profile": {"shape": {"rows": 3, "columns": 2}},
            "execution": {},
        }

    monkeypatch.setattr(analysis_api, "run_full_analysis", fake_run_full_analysis)
    analysis_api.analyze(
        pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]}),
        mode="deep",
        target="target",
        artifacts=False,
    )

    disabled = set(captured["disabled_modules"])
    assert disabled == {"modeling", "explainability"}
    assert "deep_statistics" not in disabled
    assert "text_profile" not in disabled


def test_research_keeps_modeling_and_explainability_enabled(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_full_analysis(**kwargs):
        captured.update(kwargs)
        return {
            "filename": "<dataframe>",
            "profile": {"shape": {"rows": 3, "columns": 2}},
            "execution": {},
        }

    monkeypatch.setattr(analysis_api, "run_full_analysis", fake_run_full_analysis)
    analysis_api.analyze(
        pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]}),
        mode="research",
        target="target",
        artifacts=False,
    )

    assert tuple(captured["disabled_modules"]) == ()


def test_selector_matches_standard_deep_and_research_contracts():
    signals = {
        "row_count": 8_000,
        "has_numeric_columns": True,
        "has_multiple_numeric_columns": True,
        "has_categorical_columns": True,
        "has_long_text_columns": True,
        "has_datetime_columns": True,
        "has_time_series_structure": True,
        "has_id_like_columns": True,
        "has_high_missingness": True,
        "has_sensitive_column_candidates": True,
        "has_email_like_columns": True,
    }

    standard = select_analyses(signals, analysis_mode="standard", target_column="target")
    deep = select_analyses(signals, analysis_mode="deep", target_column="target")
    research = select_analyses(signals, analysis_mode="research", target_column="target")

    standard_ids = {item["id"] for item in standard["selected_analyses"]}
    deep_ids = {item["id"] for item in deep["selected_analyses"]}
    research_ids = {item["id"] for item in research["selected_analyses"]}

    assert {"target_analysis", "time_series_signal"} <= standard_ids
    assert {"normality_tests", "chi_square_analysis", "text_analysis"} <= deep_ids
    assert {"feature_importance", "baseline_model"} <= research_ids

    assert {"normality_tests", "chi_square_analysis", "text_analysis"}.isdisjoint(standard_ids)
    assert {"feature_importance", "baseline_model"}.isdisjoint(deep_ids)
