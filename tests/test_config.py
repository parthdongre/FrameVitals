import pandas as pd
import pytest

import framevitals
from framevitals.config import AnalysisConfig, available_modules, resolve_config


def test_default_and_preset_resolution():
    assert resolve_config() == AnalysisConfig()

    quick = resolve_config(preset="quick")
    assert quick.mode == "quick"
    assert quick.workers == 2
    assert quick.artifacts is False
    assert quick.disabled_modules == ()

    ci = resolve_config(preset="ci")
    assert ci.mode == "standard"
    assert ci.workers == 2
    assert set(ci.disabled_modules) == {
        "modeling",
        "explainability",
        "charts",
        "ai",
    }


def test_toml_config_resolution_and_explicit_precedence(tmp_path):
    config = tmp_path / "framevitals.toml"
    config.write_text(
        "[analysis]\n"
        "preset = \"deep\"\n"
        "target = \"churn\"\n"
        "artifacts = true\n"
        "disabled_modules = [\"text_profile\"]\n"
        "\n"
        "[resources]\n"
        "workers = 6\n"
        "\n"
        "[modules]\n"
        "anomaly_detection = false\n",
        encoding="utf-8",
    )

    resolved = resolve_config(config)
    assert resolved.mode == "deep"
    assert resolved.target == "churn"
    assert resolved.artifacts is True
    assert resolved.workers == 6
    assert resolved.disabled_modules == ("text_profile", "anomaly_detection")

    overridden = resolve_config(
        config,
        mode="quick",
        target="label",
        artifacts=False,
        workers=3,
        disabled_modules=["charts"],
    )
    assert overridden == AnalysisConfig(
        mode="quick",
        target="label",
        artifacts=False,
        workers=3,
        disabled_modules=("charts",),
    )


def test_module_booleans_can_selectively_override_preset_defaults():
    resolved = resolve_config(
        preset="ci",
        config={
            "modules": {
                "modeling": True,
                "time_series": False,
            },
        },
    )

    assert "modeling" not in resolved.disabled_modules
    assert "time_series" in resolved.disabled_modules
    assert "explainability" in resolved.disabled_modules
    assert "charts" in resolved.disabled_modules
    assert "ai" in resolved.disabled_modules


def test_config_mapping_and_validation():
    resolved = resolve_config({
        "analysis": {"mode": "research", "artifacts": False},
        "resources": {"workers": 8},
        "modules": {"deep_statistics": False},
    })
    assert resolved.mode == "research"
    assert resolved.workers == 8
    assert resolved.disabled_modules == ("deep_statistics",)

    assert "modeling" in available_modules()
    assert "cleaning" in available_modules()

    with pytest.raises(ValueError, match="Unknown FrameVitals preset"):
        resolve_config(preset="everything")
    with pytest.raises(ValueError, match="workers"):
        resolve_config(workers=0)
    with pytest.raises(ValueError, match="Unknown FrameVitals module"):
        resolve_config(disabled_modules=["magic_model"])
    with pytest.raises(ValueError, match="Unknown FrameVitals module in"):
        resolve_config({"modules": {"magic_model": False}})


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
        "disabled_modules": (),
    }
    assert result["execution"]["disabled_modules"] == []
    assert not (tmp_path / "cleaned").exists()
