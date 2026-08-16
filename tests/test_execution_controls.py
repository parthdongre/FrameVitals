import pandas as pd

import framevitals
from framevitals.cli import build_parser, main


def _frame(rows: int = 36) -> pd.DataFrame:
    return pd.DataFrame({
        "value": list(range(rows)),
        "value_2": [index * 2 for index in range(rows)],
        "segment": ["A", "B", "C"] * (rows // 3),
        "target": [0, 1] * (rows // 2),
    })


def test_analyze_can_disable_expensive_modules_without_losing_result_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = framevitals.analyze(
        _frame(),
        target="target",
        mode="standard",
        artifacts=True,
        disabled_modules=[
            "deep_statistics",
            "anomaly_detection",
            "time_series",
            "text_profile",
            "target_intelligence",
            "modeling",
            "explainability",
            "cleaning",
            "charts",
            "ai",
        ],
    )

    assert result["execution"]["disabled_modules"] == sorted(
        framevitals.available_modules()
    )
    assert all(
        status == "disabled"
        for status in result["execution"]["module_status"].values()
    )

    assert result["deep_statistics_v2"]["skipped"] is True
    assert result["anomalies_v2"]["skipped"] is True
    assert result["time_series"]["skipped"] is True
    assert result["text_profile"]["skipped"] is True
    assert result["target_intelligence"]["skipped"] is True
    assert result["model_leaderboard"]["skipped"] is True
    assert result["explainability"]["skipped"] is True
    assert result["cleaning"]["skipped"] is True
    assert result["charts"] == []
    assert result["ai_report"]["source"] == "disabled"

    assert not (tmp_path / "cleaned").exists()
    assert not (tmp_path / "static" / "charts").exists()


def test_ci_preset_skips_modeling_but_keeps_target_intelligence():
    result = framevitals.analyze(
        _frame(),
        target="target",
        preset="ci",
    )

    execution = result["execution"]["module_status"]
    assert execution["modeling"] == "disabled"
    assert execution["explainability"] == "disabled"
    assert execution["charts"] == "disabled"
    assert execution["ai"] == "disabled"
    assert execution["target_intelligence"] == "ran"

    assert result["model_leaderboard"]["skipped"] is True
    assert result["target_intelligence"]["available"] is True


def test_plan_surfaces_execution_module_selection():
    plan = framevitals.plan(
        _frame(),
        target="target",
        mode="deep",
        disabled_modules=["modeling", "charts"],
    )

    modules = plan["selection"]["execution_modules"]
    assert modules["disabled"] == ["charts", "modeling"]
    assert "anomaly_detection" in modules["enabled"]
    assert plan["config"]["disabled_modules"] == ("modeling", "charts")


def test_cli_accepts_repeatable_disable_module_flags():
    parser = build_parser()
    args = parser.parse_args([
        "analyze",
        "data.csv",
        "--disable-module",
        "modeling",
        "--disable-module",
        "charts",
    ])

    assert args.disabled_modules == ["modeling", "charts"]


def test_cli_module_flags_reach_pipeline(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "data.csv"
    _frame().to_csv(dataset, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "framevitals",
            "analyze",
            str(dataset),
            "--mode",
            "standard",
            "--disable-module",
            "anomaly_detection",
            "--disable-module",
            "modeling",
            "--format",
            "json",
        ],
    )

    assert main() == 0
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["execution"]["module_status"]["anomaly_detection"] == "disabled"
    assert payload["anomalies_v2"]["skipped"] is True

    # Modeling requires a target in the first place. The configuration still
    # records it as disabled, but execution correctly explains that this run
    # skipped it because it was not applicable.
    assert "modeling" in payload["execution"]["disabled_modules"]
    assert payload["execution"]["module_status"]["modeling"] == "not_applicable"
