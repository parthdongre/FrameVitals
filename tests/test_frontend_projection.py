from pathlib import Path

import pandas as pd

import framevitals.frontend_api as frontend_api


def test_dashboard_payload_projects_canonical_prism_result_without_reanalysis(tmp_path):
    df = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1],
    })
    leakage = {
        "available": True,
        "target_column": "target",
        "status": "Review Recommended",
        "warnings": [
            {
                "feature": "feature",
                "risk": "Medium",
                "reason": "example",
                "correlation": 0.5,
            }
        ],
    }
    target_profile = {
        "available": True,
        "target_column": "target",
        "task_type": "classification",
    }
    target_intelligence = {
        "available": True,
        "target_column": "target",
        "task_type": "classification",
        "target_profile": target_profile,
        "leakage": leakage,
        "top_associations": [
            {"feature": "feature", "score": 0.5, "method": "point_biserial"},
        ],
    }
    explainability = {
        "available": True,
        "global_importance": [{"feature": "feature", "importance": 0.7}],
    }

    result = {
        "dataset_id": "single-pass",
        "filename": "data.csv",
        "profile": {
            "shape": {"rows": 4, "columns": 2},
            "numeric_columns": ["feature", "target"],
            "categorical_columns": [],
            "date_columns": [],
        },
        "column_roles": {
            "feature": {"roles": ["numeric"]},
            "target": {"roles": ["target_candidate"]},
        },
        "roles_summary": {"target_candidates": ["target"]},
        "analysis_selection": {"summary": {}},
        "health": {"overall_score": 90, "label": "healthy"},
        "ml_readiness": {"score": 80, "label": "ready"},
        "target_intelligence": target_intelligence,
        "explainability": explainability,
        "quality_diagnostics": {"available": True},
        "deep_statistics_v2": {"available": True},
        "anomalies_v2": {"available": False},
        "model_leaderboard": {"available": False},
        "time_series": {"available": False},
        "text_profile": {"available": False},
        "cleaning": {},
        "charts": [],
        "signals": [],
        "dataset_signals": {},
        "advanced": {},
        "ai_report": {},
        "timings_ms": {"total": 1.0},
    }

    payload = frontend_api.build_dashboard_payload(
        result,
        df,
        Path(tmp_path / "data.csv"),
        "standard",
        1.0,
        target_column="target",
    )

    assert payload["targetIntelligence"] is target_intelligence
    assert payload["targetAnalysis"] is target_profile
    assert payload["targetLeakage"] is leakage
    assert payload["featureImportance"] is explainability
    assert payload["deepStatistics"] is None
    assert payload["baselineModel"]["available"] is False
    assert payload["modelDiagnostics"]["available"] is False


def test_frontend_api_does_not_import_legacy_analysis_runners():
    for name in (
        "run_deep_statistics",
        "analyze_target",
        "run_feature_importance",
        "run_baseline_model",
        "run_target_leakage_analysis",
        "run_multicollinearity_analysis",
        "run_predictive_diagnostics",
        "run_segment_analysis",
    ):
        assert name not in frontend_api.__dict__
