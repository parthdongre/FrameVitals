import pandas as pd

from framevitals.advanced_indicators import (
    calculate_advanced_indicators,
)
from framevitals.analysis_selector import (
    select_analyses,
)
from framevitals.cleaner import (
    create_cleaned_dataset,
)
from framevitals.column_roles import (
    infer_column_roles,
)
from framevitals.dataset_signals import (
    detect_dataset_signals,
)
from framevitals.health_score import (
    calculate_health_score,
)
from framevitals.ml_readiness import (
    calculate_ml_readiness,
)
from framevitals.profiler import (
    build_profile,
)
from framevitals.signal_engine import (
    build_signals,
)


def make_dataset():
    return pd.DataFrame({
        "age": [20, 21, 22, 23, 24],
        "income": [
            30000,
            32000,
            35000,
            37000,
            40000,
        ],
        "city": [
            "Pune",
            "Mumbai",
            "Pune",
            "Nashik",
            "Mumbai",
        ],
    })


def test_analysis_selector():
    df = make_dataset()

    profile = build_profile(df)

    dataset_signals = detect_dataset_signals(
        df,
        profile,
    )

    selection = select_analyses(
        dataset_signals,
        analysis_mode="quick",
    )

    assert selection["summary"]["selected_count"] > 0

    selected_ids = {
        item["id"]
        for item in selection["selected_analyses"]
    }

    assert "ingestion_analysis" in selected_ids
    assert "structural_profile" in selected_ids


def test_dataset_signals_cached_roles_match_standalone():
    df = pd.DataFrame({
        "customer_id": ["A1", "A2", "A3", "A3"],
        "email": [
            "a@example.com",
            "b@example.com",
            None,
            None,
        ],
        "notes": [
            "x" * 220,
            "short",
            "another short note",
            "another short note",
        ],
        "value": [1.0, 2.0, 3.0, 3.0],
    })
    profile = build_profile(df)
    roles = infer_column_roles(df)

    standalone = detect_dataset_signals(df, profile)
    cached = detect_dataset_signals(df, profile, column_roles=roles)

    assert cached == standalone


def test_display_signals():
    df = make_dataset()

    profile = build_profile(df)

    health = calculate_health_score(
        df,
        profile,
    )

    readiness = calculate_ml_readiness(df)
    advanced = calculate_advanced_indicators(df)

    signals = build_signals(
        profile,
        health,
        readiness,
        advanced,
    )

    assert isinstance(signals, list)
    assert len(signals) > 0

    names = {
        signal["name"]
        for signal in signals
    }

    assert "Data Completeness" in names
    assert "ML Readiness" in names


def test_ml_readiness_cached_profile_matches_standalone():
    df = pd.DataFrame({
        "age": [20, None, 22, 22],
        "city": ["Pune", "Mumbai", None, None],
    })
    profile = build_profile(df)

    assert calculate_ml_readiness(df, profile=profile) == calculate_ml_readiness(df)


def test_cleaner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = pd.DataFrame({
        "age": [
            20,
            None,
            30,
        ],
        "city": [
            "Pune",
            "Mumbai",
            None,
        ],
    })

    result = create_cleaned_dataset(
        "test_dataset",
        df,
    )

    assert result["missing_before"] == 2
    assert result["missing_after"] == 0

    cleaned_path = (
        tmp_path
        / result["output_path"]
    )

    assert cleaned_path.exists()


def test_cleaner_cached_inputs_match_standalone(tmp_path):
    df = pd.DataFrame({
        "age": [20, None, 30, 30],
        "city": ["Pune", "Mumbai", None, None],
    })
    profile = build_profile(df)
    health = calculate_health_score(df, profile)

    standalone = create_cleaned_dataset(
        "standalone",
        df,
        write_output=False,
    )
    cached = create_cleaned_dataset(
        "cached",
        df,
        write_output=False,
        before_profile=profile,
        before_health=health,
    )

    assert cached == standalone
