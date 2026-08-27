from framevitals import ai_insights


def _context():
    profile = {
        "shape": {"rows": 10, "columns": 1},
        "columns": ["x"],
        "dtypes": {"x": "int64"},
        "missing_counts": {"x": 0},
        "duplicate_rows": 0,
        "numeric_columns": ["x"],
        "categorical_columns": [],
        "date_columns": [],
    }
    health = {"overall_score": 90, "label": "Good", "details": {}}
    readiness = {"score": 80, "label": "Ready", "recommendations": []}
    return profile, health, readiness


def test_ai_report_fallback_preserves_endpoint_errors(monkeypatch):
    profile, health, readiness = _context()

    def fail_openrouter(*args, **kwargs):
        raise RuntimeError("openrouter down")

    def fail_ollama(*args, **kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(ai_insights, "_call_openrouter", fail_openrouter)
    monkeypatch.setattr(ai_insights, "_call_ollama", fail_ollama)

    result = ai_insights.generate_ai_report(
        profile,
        health,
        [],
        readiness,
        advanced={},
    )

    assert result["source"].startswith("fallback:")
    assert "openrouter down" in result["source"]
    assert "ollama down" in result["source"]


def test_ai_settings_are_read_at_call_time(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "first-model")
    assert ai_insights._openrouter_model() == "first-model"
    monkeypatch.setenv("OPENROUTER_MODEL", "second-model")
    assert ai_insights._openrouter_model() == "second-model"
