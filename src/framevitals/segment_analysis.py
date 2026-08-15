import pandas as pd
import numpy as np


def choose_segment_columns(df, max_unique=12):
    candidates = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        unique_count = df[column].nunique(dropna=True)

        if 2 <= unique_count <= max_unique:
            candidates.append(column)

    return candidates[:5]


def run_segment_analysis(df, target_column=None):
    segment_columns = choose_segment_columns(df)

    if not segment_columns:
        return {
            "available": False,
            "message": "No suitable low-cardinality segment columns detected.",
        }

    results = []

    for segment in segment_columns:
        item = {
            "segment_column": segment,
            "groups": [],
        }

        if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
            grouped = df.groupby(segment, dropna=False)[target_column].agg(
                ["count", "mean", "median", "std", "min", "max"]
            ).reset_index()

            for _, row in grouped.iterrows():
                item["groups"].append(
                    {
                        "group": str(row[segment]),
                        "count": int(row["count"]),
                        "target_mean": None if pd.isna(row["mean"]) else round(float(row["mean"]), 4),
                        "target_median": None if pd.isna(row["median"]) else round(float(row["median"]), 4),
                        "target_std": None if pd.isna(row["std"]) else round(float(row["std"]), 4),
                        "target_min": None if pd.isna(row["min"]) else round(float(row["min"]), 4),
                        "target_max": None if pd.isna(row["max"]) else round(float(row["max"]), 4),
                    }
                )

            item["analysis_type"] = "numeric_target_by_segment"

        else:
            counts = df[segment].value_counts(dropna=False)
            total = max(len(df), 1)

            for value, count in counts.items():
                item["groups"].append(
                    {
                        "group": str(value),
                        "count": int(count),
                        "percent": round(float(count / total * 100), 2),
                    }
                )

            item["analysis_type"] = "segment_distribution"

        results.append(item)

    return {
        "available": True,
        "segment_columns": segment_columns,
        "results": results,
        "interpretation": (
            "Segment analysis compares dataset behaviour across low-cardinality categorical groups. "
            "It is useful for identifying group imbalance, source-specific patterns, or target differences."
        ),
    }
