import numpy as np
import pandas as pd

SENSITIVE_KEYWORDS = [
    "gender", "sex", "age", "race", "religion",
    "region", "location", "income", "salary", "caste",
]


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
    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        return {"anomalous_rows": 0, "highest_score": 0, "top_rows": []}

    flags = pd.DataFrame(index=df.index)

    for col in numeric.columns:
        q1 = numeric[col].quantile(0.25)
        q3 = numeric[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            flags[col] = False
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flags[col] = (numeric[col] < lower) | (numeric[col] > upper)

    scores = flags.sum(axis=1) / max(len(numeric.columns), 1)
    top = scores.sort_values(ascending=False).head(10)

    return {
        "anomalous_rows": int((scores > 0).sum()),
        "highest_score": round(float(scores.max()), 3),
        "top_rows": [
            {"row_index": int(idx), "score": round(float(score), 3)}
            for idx, score in top.items()
            if score > 0
        ],
    }


def detect_fairness_review(df):
    found = []

    for col in df.columns:
        lower_col = col.lower()
        if any(keyword in lower_col for keyword in SENSITIVE_KEYWORDS):
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
    date_columns = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")

        if parsed.notna().mean() >= 0.7:
            date_columns.append((col, parsed))

    if not date_columns:
        return {"available": False, "message": "No strong date column detected."}

    col, parsed = date_columns[0]
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
