import json

import pandas as pd

import framevitals
from framevitals.cleaner import create_cleaned_dataset
from framevitals.cleaning_plan import infer_cleaning_plan
from framevitals.cli import main
from framevitals.health_score import calculate_health_score
from framevitals.profiler import build_profile


def _dirty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [20.0, None, 30.0, 30.0],
        "city": ["Pune", "Mumbai", None, None],
    })


def test_cleaning_plan_is_explicit_and_does_not_mutate_input():
    df = _dirty_frame()
    original = df.copy(deep=True)

    plan = infer_cleaning_plan(df)

    pd.testing.assert_frame_equal(df, original)
    assert plan["schema_version"] == "1"
    assert plan.summary()["action_count"] == 3
    assert plan["duplicates_to_remove"] == 1
    assert plan["missing_values_to_fill"] == 2
    assert [action["type"] for action in plan.actions] == [
        "remove_duplicates",
        "fill_numeric_missing",
        "fill_categorical_missing",
    ]


def test_apply_cleaning_plan_returns_clean_copy():
    df = _dirty_frame()
    plan = infer_cleaning_plan(df)

    cleaned = plan.apply(df)

    assert cleaned is not df
    assert len(cleaned) == 3
    assert int(cleaned.isna().sum().sum()) == 0
    assert cleaned.loc[1, "age"] == 25.0
    assert cleaned.loc[2, "city"] == "Mumbai"
    assert pd.isna(df.loc[1, "age"])


def test_cleaning_simulation_reports_expected_improvement():
    df = _dirty_frame()
    profile = build_profile(df)
    health = calculate_health_score(df, profile)
    plan = infer_cleaning_plan(df, profile=profile)

    simulation = plan.simulate(
        df,
        before_profile=profile,
        before_health=health,
    )

    assert simulation["rows_removed"] == 1
    assert simulation["missing_before"] == 3
    assert simulation["missing_after"] == 0
    assert simulation["duplicates_before"] == 1
    assert simulation["duplicates_after"] == 0
    assert simulation["health_delta"] >= 0


def test_internal_cleaner_preserves_legacy_payload_and_adds_structured_plan():
    df = _dirty_frame()
    result = create_cleaned_dataset("test", df, write_output=False)

    assert result["actions"] == [
        {
            "action": "Remove duplicates",
            "details": "Removed 1 duplicate rows.",
            "risk": "Low",
        },
        {
            "action": "Fill numeric missing values",
            "details": "Filled 1 missing values in 'age' using median.",
            "risk": "Medium",
        },
        {
            "action": "Fill categorical missing values",
            "details": "Filled 1 missing values in 'city' using mode.",
            "risk": "Medium",
        },
    ]
    assert result["missing_before"] == 3
    assert result["missing_after"] == 0
    assert result["duplicates_before"] == 1
    assert result["duplicates_after"] == 0
    assert result["plan"]["schema_version"] == "1"


def test_public_cleaning_api_supports_dataframe_and_file(tmp_path):
    df = _dirty_frame()
    plan = framevitals.plan_cleaning(df)
    cleaned = framevitals.clean(df, plan=plan)

    assert isinstance(plan, framevitals.CleaningPlan)
    assert len(cleaned) == 3
    assert cleaned.isna().sum().sum() == 0

    path = tmp_path / "dirty.csv"
    df.to_csv(path, index=False)
    file_plan = framevitals.plan_cleaning(path)
    file_cleaned = framevitals.clean(path, plan=file_plan)

    assert file_plan.summary()["action_count"] == 3
    assert len(file_cleaned) == 3
    assert file_cleaned.isna().sum().sum() == 0


def test_datetime_mode_fill_stays_datetime_dtype():
    df = pd.DataFrame({
        "event_time": [
            pd.Timestamp("2026-01-01"),
            pd.NaT,
            pd.Timestamp("2026-01-01"),
        ],
    })
    plan = infer_cleaning_plan(df)
    cleaned = plan.apply(df)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["event_time"])
    assert cleaned["event_time"].isna().sum() == 0


def test_cli_clean_is_plan_only_without_output_and_explicit_with_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    dataset = tmp_path / "dirty.csv"
    cleaned_path = tmp_path / "cleaned.csv"
    plan_path = tmp_path / "plan.json"
    _dirty_frame().to_csv(dataset, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["framevitals", "clean", str(dataset), "--plan-output", str(plan_path)],
    )
    assert main() == 0
    plan_only_payload = json.loads(capsys.readouterr().out)
    assert plan_only_payload["cleaned_output"] is None
    assert not cleaned_path.exists()
    assert plan_path.exists()

    monkeypatch.setattr(
        "sys.argv",
        ["framevitals", "clean", str(dataset), "--output", str(cleaned_path)],
    )
    assert main() == 0
    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_payload["cleaned_output"] == str(cleaned_path)
    assert cleaned_path.exists()

    cleaned = pd.read_csv(cleaned_path)
    assert len(cleaned) == 3
    assert cleaned.isna().sum().sum() == 0
