from __future__ import annotations

import re

import numpy as np
import pandas as pd

SENSITIVE_KEYWORDS = {
    "gender",
    "sex",
    "age",
    "race",
    "religion",
    "region",
    "location",
    "income",
    "salary",
    "caste",
}


def _column_tokens(name: object) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return {token for token in normalized.split("_") if token}


def _bounded_non_null_sample(series: pd.Series, max_rows: int = 200) -> pd.Series:
    clean = series.dropna()
    if len(clean) <= max_rows:
        return clean
    positions = np.linspace(0, len(clean) - 1, num=max_rows, dtype=np.int64)
    return clean.iloc[np.unique(positions)]


def calculate_column_utility(df):
    rows = max(len(df), 1)
    results = []

    for col in df.columns:
        missing_percent = df[col].isna().mean() * 100
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / rows

        score = 100
        score -= min(40, missing_percent)

        if unique_count <= 1:
            score -= 40
        if unique_ratio > 0.9 and not pd.api.types.is_numeric_dtype(df[col]):
            score -= 30

        score = round(max(0, min(100, score)), 2)

        if score >= 80:
            status = "High"
        elif score >= 50:
            status = "Moderate"
        else:
            status = "Low"

        results.append({
            "column": col,
            "score": score,
            "status": status,
            "missing_percent": round(missing_percent, 2),
            "unique_count": int(unique_count),
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


def calculate_anomalies(df):
    """Calculate simple IQR anomaly density using O(rows) auxiliary memory."""
    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        return {"anomalous_rows": 0, "highest_score": 0, "top_rows": []}

    # The old implementation materialized one boolean column per numeric
    # feature. A single counter vector produces the identical row score while
    # avoiding O(rows * numeric_columns) temporary memory.
    row_hits = np.zeros(len(df), dtype=np.uint32)
    usable_columns = 0

    for col in numeric.columns:
        series = numeric[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = ((series < lower) | (series > upper)).to_numpy(dtype=bool)
        row_hits += mask
        usable_columns += 1

    denominator = max(usable_columns, 1)
    scores = row_hits.astype(np.float64) / denominator
    anomalous_mask = row_hits > 0

    if not anomalous_mask.any():
        return {"anomalous_rows": 0, "highest_score": 0, "top_rows": []}

    top_count = min(10, len(scores))
    candidate_positions = np.argpartition(scores, -top_count)[-top_count:]
    candidate_positions = candidate_positions[
        np.argsort(scores[candidate_positions])[::-1]
    ]

    top_rows = []
    for position in candidate_positions:
        score = float(scores[position])
        if score <= 0:
            continue
        index_value = df.index[int(position)]
        top_rows.append({
            "row_index": (
                int(index_value)
                if isinstance(index_value, (int, np.integer))
                else str(index_value)
            ),
            "score": round(score, 3),
        })

    return {
        "anomalous_rows": int(anomalous_mask.sum()),
        "highest_score": round(float(scores.max()), 3),
        "top_rows": top_rows,
    }


def detect_fairness_review(df):
    found = []

    for col in df.columns:
        tokens = _column_tokens(col)
        if tokens & SENSITIVE_KEYWORDS:
            found.append(col)

    if found:
        return {
            "needs_review": True,
            "columns": found,
            "message": "Potential demographic or sensitive columns detected: " + ", ".join(found),
        }

    return {
        "needs_review": False,
        "columns": [],
        "message": "No obvious demographic-like columns detected.",
    }


def calculate_freshness(df):
    """Detect a date column with bounded screening and one full parse at most."""
    candidates: list[tuple[float, str]] = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue

        sample = _bounded_non_null_sample(series, 200)
        if sample.empty:
            continue

        parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
        parse_rate = float(parsed_sample.notna().mean())
        if parse_rate >= 0.7:
            candidates.append((parse_rate, col))

    if not candidates:
        return {"available": False, "message": "No strong date column detected."}

    # Parse only the strongest candidate across the complete column. Future
    # Arrow/Rust sources will calculate extrema while streaming instead.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    _, col = candidates[0]
    parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
    if parsed.notna().mean() < 0.7:
        return {"available": False, "message": "No strong date column detected."}

    min_date = parsed.min()
    max_date = parsed.max()

    return {
        "available": True,
        "date_column": col,
        "oldest_record": str(min_date.date()) if pd.notna(min_date) else None,
        "latest_record": str(max_date.date()) if pd.notna(max_date) else None,
        "message": f"Date coverage detected using column '{col}'.",
    }


def calculate_advanced_indicators(df):
    return {
        "column_utility": calculate_column_utility(df),
        "anomalies": calculate_anomalies(df),
        "fairness": detect_fairness_review(df),
        "freshness": calculate_freshness(df),
    }
