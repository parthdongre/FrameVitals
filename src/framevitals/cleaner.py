from pathlib import Path

import pandas as pd

from framevitals.health_score import calculate_health_score
from framevitals.profiler import build_profile
from framevitals.security import sanitize_csv_value

CLEANED_DIR = Path("cleaned")


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

    Library callers can set ``write_output=False`` to keep analysis free of
    filesystem side effects. Application callers retain the historical
    behavior by leaving ``write_output`` enabled.

    ``before_profile`` and ``before_health`` are optional cached values. The
    main pipeline supplies them to avoid profiling/scoring the original data a
    second time; standalone callers can omit them with unchanged behavior.
    """
    cleaned = df.copy()
    actions = []

    if before_profile is None:
        before_profile = build_profile(df)
    if before_health is None:
        before_health = calculate_health_score(df, before_profile)

    cached_duplicates = before_profile.get("duplicate_rows")
    duplicate_count = int(
        cached_duplicates if cached_duplicates is not None else df.duplicated().sum()
    )
    missing_before = sum(
        int(value)
        for value in before_profile.get("missing_counts", {}).values()
        if value is not None
    )

    if duplicate_count:
        cleaned = cleaned.drop_duplicates()
        actions.append({
            "action": "Remove duplicates",
            "details": f"Removed {duplicate_count} duplicate rows.",
            "risk": "Low",
        })

    missing_counts = cleaned.isna().sum()
    for col, missing_value in missing_counts.items():
        missing = int(missing_value)
        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[col]):
            value = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(value)
            actions.append({
                "action": "Fill numeric missing values",
                "details": f"Filled {missing} missing values in '{col}' using median.",
                "risk": "Medium",
            })
        else:
            mode = cleaned[col].mode(dropna=True)
            value = mode.iloc[0] if len(mode) else "Unknown"
            cleaned[col] = cleaned[col].fillna(value)
            actions.append({
                "action": "Fill categorical missing values",
                "details": f"Filled {missing} missing values in '{col}' using mode.",
                "risk": "Medium",
            })

    after_profile = build_profile(cleaned)
    after_health = calculate_health_score(cleaned, after_profile)
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
        "before_health": before_health,
        "after_health": after_health,
        "output_path": str(output_path) if output_path is not None else None,
        "missing_before": int(missing_before),
        "missing_after": int(missing_after),
        "duplicates_before": duplicate_count,
        "duplicates_after": int(after_profile.get("duplicate_rows", 0)),
    }
