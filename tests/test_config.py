import pandas as pd
import pytest

import framevitals
from framevitals.config import AnalysisConfig, resolve_config


def test_default_and_preset_resolution():
    assert resolve_config() == AnalysisConfig()

    quick = resolve_config(preset="quick")
    assert quick.mode == "quick"
    assert quick.workers == 2
    assert quick.artifacts is False


def test_toml_config_resolution_and_explicit_precedence(tmp_path):
    config = tmp_path / "framevitals.toml"
    config.write_text(
        "[analysis]\n"
        "preset = \"deep\"\n"
        "target = \"churn\"\n"
        "artifacts = true\n"
        "\n"
        "[resources]\n"
        "workers = 6\n",
        encoding="utf-8",
    )

    resolved = resolve_config(config)
    assert resolved.mode == "deep"
    assert resolved.target == "churn"
    assert resolved.artifacts is True
    assert resolved.workers == 6

    overridden = resolve_config(
        config,
        mode="quick",
        target="label",
        artifacts=False,
        workers=3,
    )
    assert overridden == AnalysisConfig(
        mode="quick",
        target="label",
        artifacts=False,
        workers=3,
    )


def test_config_mapping_and_validation():
    resolved = resolve_config({
        "analysis": {"mode": "research", "artifacts": False},
        "resources": {"workers": 8},
    })
    assert resolved.mode == "research"
    assert resolved.workers == 8

    with pytest.raises(ValueError, match="Unknown FrameVitals preset"):
        resolve_config(preset="everything")
    with pytest.raises(ValueError, match="workers"):
        resolve_config(workers=0)


def test_public_analyze_honors_config_and_records_resolution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "city": ["Pune", "Mumbai", "Pune", "Nashik"],
    })

    result = framevitals.analyze(
        df,
        config={
            "analysis": {"mode": "quick", "artifacts": False},
            "resources": {"workers": 1},
        },
    )

    assert result["analysis_mode"] == "quick"
    assert result["artifacts_enabled"] is False
    assert result["config"] == {
        "mode": "quick",
        "target": None,
        "artifacts": False,
        "workers": 1,
    }
    assert not (tmp_path / "cleaned").exists()
