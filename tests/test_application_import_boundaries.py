from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


MIGRATED_MODULES = {
    "advanced_indicators",
    "ai_insights",
    "analysis_selector",
    "anomaly_ensemble",
    "cleaner",
    "column_roles",
    "dataset_signals",
    "deep_statistics_v2",
    "explainability",
    "health_score",
    "loader",
    "ml_readiness",
    "model_leaderboard",
    "pipeline",
    "profiler",
    "signal_engine",
    "text_profile",
    "time_series",
    "visualizer",
}


def assert_no_migrated_module_imports(
    filename,
):
    source = (
        ROOT / filename
    ).read_text(
        encoding="utf-8"
    )

    for module in MIGRATED_MODULES:
        legacy_import = (
            f"from modules.{module} import"
        )

        assert legacy_import not in source, (
            f"{filename} still imports "
            f"{module} through the legacy "
            "modules namespace."
        )


def test_flask_app_uses_framevitals():
    assert_no_migrated_module_imports(
        "app.py"
    )


def test_streamlit_app_uses_framevitals():
    assert_no_migrated_module_imports(
        "streamlit_app.py"
    )
    
def test_ai_agent_uses_framevitals_agent_layer():
    source = (
        ROOT / "modules" / "ai_agent.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "from modules.agent_tools import"
        not in source
    )

    assert (
        "from modules.agent_brief import"
        not in source
    )
