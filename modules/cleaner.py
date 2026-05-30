from pathlib import Path
import pandas as pd
from modules.health_score import calculate_health_score
from modules.profiler import build_profile
from modules.security import sanitize_csv_value

CLEANED_DIR = Path("cleaned")
CLEANED_DIR.mkdir(exist_ok=True)

def create_cleaned_dataset(dataset_id, df):
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

    safe_df = cleaned.map(sanitize_csv_value)
    output_path = CLEANED_DIR / f"{dataset_id}_cleaned.csv"
    safe_df.to_csv(output_path, index=False)

    return {
        "actions": actions,
        "before_health": before_health,
        "after_health": after_health,
        "output_path": str(output_path),
        "missing_before": int(df.isna().sum().sum()),
        "missing_after": int(cleaned.isna().sum().sum()),
        "duplicates_before": int(df.duplicated().sum()),
        "duplicates_after": int(cleaned.duplicated().sum()),
    }
