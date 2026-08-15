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
    "ai_agent",
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
    
def test_framevitals_ai_agent_has_no_legacy_imports():
    source = (
        ROOT
        / "src"
        / "framevitals"
        / "ai_agent.py"
    ).read_text(
        encoding="utf-8"
    )

    assert "from modules." not in source

def test_component_harness_uses_framevitals_ai_agent():
    source = (
        ROOT
        / "tools"
        / "component_test.py"
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "from modules.ai_agent import"
        not in source
    )
