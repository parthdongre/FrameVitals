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
):
    """Build a conservative cleaned copy and optionally persist it as CSV.

    Library callers can set ``write_output=False`` to keep analysis free of
    filesystem side effects. Application callers retain the historical
    behavior by leaving ``write_output`` enabled.
    """
    cleaned = df.copy()
    actions = []

    duplicate_count = int(cleaned.duplicated().sum())

    if duplicate_count:
        cleaned = cleaned.drop_duplicates()
        actions.append({
            "action": "Remove duplicates",
            "details": f"Removed {duplicate_count} duplicate rows.",
            "risk": "Low",
        })

    for col in cleaned.columns:
        missing = int(cleaned[col].isna().sum())

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

    before_profile = build_profile(df)
    before_health = calculate_health_score(df, before_profile)

    after_profile = build_profile(cleaned)
    after_health = calculate_health_score(cleaned, after_profile)

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
        "missing_before": int(df.isna().sum().sum()),
        "missing_after": int(cleaned.isna().sum().sum()),
        "duplicates_before": int(df.duplicated().sum()),
        "duplicates_after": int(cleaned.duplicated().sum()),
    }
