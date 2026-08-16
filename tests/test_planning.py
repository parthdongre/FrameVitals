import json

import pandas as pd

import framevitals
from framevitals.cli import main


def _dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "age": list(range(20, 40)),
        "income": [30000 + index * 1000 for index in range(20)],
        "city": ["Pune", "Mumbai"] * 10,
        "churn": [0, 1] * 10,
    })


def test_public_plan_returns_explainable_dict_compatible_plan():
    result = framevitals.plan(_dataset(), mode="quick")

    assert isinstance(result, framevitals.AnalysisPlan)
    assert isinstance(result, dict)
    assert result["dataset_name"] == "<dataframe>"
    assert result.summary()["selected_count"] > 0
    assert any(item["id"] == "structural_profile" for item in result.selected)
    assert "FrameVitals Analysis Plan" in result.explain_text()
    assert "preview only" in result.explain_text().lower()


def test_deep_plan_keeps_target_analysis_and_reserves_modeling_for_research():
    deep = framevitals.plan(_dataset(), mode="deep", target="churn")
    deep_ids = {item["id"] for item in deep.selected}

    assert "target_analysis" in deep_ids
    assert "baseline_model" not in deep_ids
    assert "feature_importance" not in deep_ids
    assert deep["config"]["target"] == "churn"

    research = framevitals.plan(_dataset(), mode="research", target="churn")
    research_ids = {item["id"] for item in research.selected}

    assert {"target_analysis", "baseline_model", "feature_importance"} <= research_ids
    assert research["config"]["target"] == "churn"


def test_plan_uses_config_without_writing_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "framevitals.toml"
    config.write_text(
        "[analysis]\nmode = \"deep\"\ntarget = \"churn\"\nartifacts = true\n"
        "[resources]\nworkers = 1\n",
        encoding="utf-8",
    )

    result = framevitals.plan(_dataset(), config=config)

    assert result["analysis_mode"] == "deep"
    assert result["target"] == "churn"
    assert result["config"]["artifacts"] is False
    assert not (tmp_path / "cleaned").exists()
    assert not (tmp_path / "static").exists()


def test_cli_plan_can_emit_json(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset.csv"
    _dataset().to_csv(dataset, index=False)
    output = tmp_path / "plan.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "plan",
            str(dataset),
            "--mode",
            "deep",
            "--target",
            "churn",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))

    assert stdout_payload["analysis_mode"] == "deep"
    assert file_payload["target"] == "churn"
    assert file_payload["selection"]["summary"]["selected_count"] > 0
