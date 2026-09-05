from framevitals.config import AnalysisConfig, resolve_config


def test_analysis_config_object_can_clear_environment_resource_caps(monkeypatch):
    monkeypatch.setenv("FRAMEVITALS_MAX_SAMPLE_ROWS", "250")
    monkeypatch.setenv("FRAMEVITALS_MAX_RELATIONSHIP_PAIRS", "7")

    resolved = resolve_config(AnalysisConfig(mode="deep", workers=3))

    assert resolved.mode == "deep"
    assert resolved.workers == 3
    assert resolved.max_sample_rows is None
    assert resolved.max_relationship_pairs is None
