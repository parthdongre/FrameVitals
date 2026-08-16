from pathlib import Path

import pandas as pd

from framevitals.cleaning_plan import apply_cleaning_plan, infer_cleaning_plan
from framevitals.health_score import calculate_health_score
from framevitals.profiler import build_profile
from framevitals.security import sanitize_csv_value

CLEANED_DIR = Path("cleaned")


def _legacy_action_view(action: dict) -> dict:
    """Keep the existing cleaner action payload stable for current callers."""
    action_type = action.get("type")
    affected = int(action.get("affected", 0))
    column = action.get("column")

    if action_type == "remove_duplicates":
        return {
            "action": "Remove duplicates",
            "details": f"Removed {affected} duplicate rows.",
            "risk": "Low",
        }
    if action_type == "fill_numeric_missing":
        return {
            "action": "Fill numeric missing values",
            "details": f"Filled {affected} missing values in '{column}' using median.",
            "risk": "Medium",
        }
    if action_type == "fill_categorical_missing":
        return {
            "action": "Fill categorical missing values",
            "details": f"Filled {affected} missing values in '{column}' using mode.",
            "risk": "Medium",
        }
    return {
        "action": str(action_type or "Cleaning action"),
        "details": str(action.get("description") or ""),
        "risk": str(action.get("risk") or "Unknown").title(),
    }


def create_cleaned_dataset(
    dataset_id,
    df,
    *,
    write_output: bool = True,
    output_dir: str | Path | None = None,
    before_profile: dict | None = None,
    before_health: dict | None = None,
):
    """Build a conservative cleaned copy and optionally persist it as CSV.

    The implementation now uses the public cleaning-plan primitives internally,
    but preserves the historical ``actions``/health/count payload for backward
    compatibility. The structured plan is returned additionally under ``plan``.
    """
    if before_profile is None:
        before_profile = build_profile(df)
    if before_health is None:
        before_health = calculate_health_score(df, before_profile)

    plan = infer_cleaning_plan(df, profile=before_profile)
    cleaned = apply_cleaning_plan(df, plan, copy=True)
    actions = [_legacy_action_view(action) for action in plan.actions]

    after_profile = build_profile(cleaned)
    after_health = calculate_health_score(cleaned, after_profile)
    missing_before = sum(
        int(value)
        for value in before_profile.get("missing_counts", {}).values()
        if value is not None
    )
    missing_after = sum(
        int(value)
        for value in after_profile.get("missing_counts", {}).values()
        if value is not None
    )

    output_path: Path | None = None
    if write_output:
        destination = Path(output_dir) if output_dir is not None else CLEANED_DIR
        destination.mkdir(parents=True, exist_ok=True)
        safe_df = cleaned.map(sanitize_csv_value)
        output_path = destination / f"{dataset_id}_cleaned.csv"
        safe_df.to_csv(output_path, index=False)

    return {
        "actions": actions,
        "plan": dict(plan),
        "before_health": before_health,
        "after_health": after_health,
        "output_path": str(output_path) if output_path is not None else None,
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "duplicates_before": int(plan.get("duplicates_to_remove", 0)),
        "duplicates_after": int(after_profile.get("duplicate_rows", 0)),
    }
