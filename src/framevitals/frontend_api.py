from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd



def format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(value, 0))
    index = 0

    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1

    if size >= 100:
        return f"{size:.0f} {units[index]}"
    if size >= 10:
        return f"{size:.1f} {units[index]}"
    return f"{size:.2f} {units[index]}"


def format_elapsed_label(elapsed_ms: float) -> str:
    total_seconds = max(float(elapsed_ms) / 1000.0, 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    if hours:
        return f"T+{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    return f"T+{minutes:02d}:{seconds:05.2f}"


def _format_edge(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value:,.0f}"
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build_numeric_distribution(df: pd.DataFrame, column: str) -> dict:
    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        return _build_empty_distribution(column)

    bin_count = min(12, max(6, int(math.sqrt(len(series)))))
    counts, edges = np.histogram(series, bins=bin_count)
    points = []

    for index, count in enumerate(counts):
        left = _format_edge(float(edges[index]))
        right = _format_edge(float(edges[index + 1]))
        points.append({"label": f"{left}-{right}", "value": int(count)})

    return {
        "title": f"{column.replace('_', ' ').title()} Distribution",
        "subtitle": f"Histogram of {column} across {len(series):,} observed values.",
        "points": points,
        "mean": round(float(series.mean()), 3),
        "stdDev": round(float(series.std(ddof=0)) if len(series) > 1 else 0.0, 3),
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "totalRows": int(len(df)),
    }


def _build_categorical_distribution(df: pd.DataFrame, column: str) -> dict:
    counts = df[column].astype(str).value_counts(dropna=False).head(12)

    if counts.empty:
        return _build_empty_distribution(column)

    values = counts.astype(float)

    return {
        "title": f"{column.replace('_', ' ').title()} Frequency Distribution",
        "subtitle": f"Top categories in {column} across {len(df):,} records.",
        "points": [{"label": str(label), "value": int(value)} for label, value in counts.items()],
        "mean": round(float(values.mean()), 3),
        "stdDev": round(float(values.std(ddof=0)) if len(values) > 1 else 0.0, 3),
        "min": round(float(values.min()), 3),
        "max": round(float(values.max()), 3),
        "totalRows": int(len(df)),
    }


def _build_empty_distribution(column: str) -> dict:
    return {
        "title": f"{column.replace('_', ' ').title()} Distribution",
        "subtitle": "No usable values were found for this quadrant.",
        "points": [{"label": "No signal", "value": 1}],
        "mean": 1.0,
        "stdDev": 0.0,
        "min": 1.0,
        "max": 1.0,
        "totalRows": 0,
    }


def build_dashboard_payload(
    result: dict,
    df: pd.DataFrame,
    file_path: Path,
    analysis_mode: str,
    elapsed_ms: float,
    target_column: str | None = None,
) -> dict:
    profile = result.get("profile", {})
    column_roles = result.get("column_roles", {})
    analysis_selection = result.get("analysis_selection", {})
    selection_summary = analysis_selection.get("summary", {})
    target_candidates = list(result.get("roles_summary", {}).get("target_candidates", []))
    selected_target_column = target_column or None

    numeric_count = len(profile.get("numeric_columns", []))
    categorical_count = len(profile.get("categorical_columns", []))
    temporal_count = len(profile.get("date_columns", []))
    boolean_count = sum(
        1 for info in column_roles.values()
        if "boolean" in info.get("roles", [])
    )

    # Pick a presentation column from the canonical profile rather than
    # requiring the complete internal role schema. The web adapter should be
    # able to project any valid AnalysisResult without re-running/inferencing
    # analysis metadata.
    profile_numeric = [
        str(column)
        for column in profile.get("numeric_columns", [])
        if str(column) in df.columns
    ]
    meaningful_numeric = []
    for column in profile_numeric:
        role_info = column_roles.get(column, {})
        roles = set(role_info.get("roles", [])) if isinstance(role_info, dict) else set()
        if not roles.intersection({"id_like", "constant"}):
            meaningful_numeric.append(column)
    distribution_column = meaningful_numeric[0] if meaningful_numeric else (
        profile_numeric[0] if profile_numeric else None
    )

    if distribution_column is not None:
        distribution = _build_numeric_distribution(df, distribution_column)
    elif profile.get("categorical_columns"):
        distribution = _build_categorical_distribution(df, profile["categorical_columns"][0])
    else:
        distribution = _build_empty_distribution("telemetry")

    # Project the canonical Prism result into the web payload. The frontend
    # adapter must never re-run statistics, fitting, leakage, VIF, or segment
    # analysis merely because a UI is being rendered.
    target_intelligence = result.get("target_intelligence") or {}
    target_analysis = (
        target_intelligence.get("target_profile", {})
        if isinstance(target_intelligence, dict)
        else {}
    )
    target_leakage = (
        target_intelligence.get("leakage", {})
        if isinstance(target_intelligence, dict)
        else {}
    )

    deep_statistics_v2 = result.get("deep_statistics_v2")
    anomalies_v2 = result.get("anomalies_v2")
    model_leaderboard = result.get("model_leaderboard")
    explainability = result.get("explainability")
    time_series_analysis = result.get("time_series")
    text_profile = result.get("text_profile")
    quality_diagnostics = result.get("quality_diagnostics") or {}

    # Legacy compatibility keys are projections only. They intentionally do
    # not trigger a second analytical pipeline.
    deep_statistics = None
    feature_importance = (
        explainability
        if isinstance(explainability, dict) and explainability.get("available")
        else {"available": False, "message": "Use canonical explainability output."}
    )
    baseline_model = {
        "available": False,
        "message": "Baseline/model comparisons are represented by modelLeaderboard.",
    }
    model_diagnostics = result.get("predictive_diagnostics") or {
        "available": False,
        "message": "No separate predictive-diagnostics pass was requested.",
    }
    multicollinearity = result.get("multicollinearity") or {}
    segment_analysis = result.get("segment_analysis") or {}

    health_score = result.get("health", {}).get("overall_score", 0)
    health_label = result.get("health", {}).get("label", "Unknown")
    result.get("ml_readiness", {}).get("score", 0)
    result.get("ml_readiness", {}).get("label", "Unknown")

    payload = {
        "id": result.get("dataset_id"),
        "filename": result.get("filename", file_path.name),
        "analysisMode": analysis_mode,
        "rows": int(profile.get("shape", {}).get("rows", 0)),
        "columns": int(profile.get("shape", {}).get("columns", 0)),
        "fileSize": format_bytes(file_path.stat().st_size if file_path.exists() else 0),
        "lastScan": format_elapsed_label(elapsed_ms),
        "selectedTargetColumn": selected_target_column,
        "targetCandidates": target_candidates,
        "profile": profile,
        "rolesSummary": result.get("roles_summary", {}),
        "columnRoles": column_roles,
        "dataTypes": [
            {"label": "Numeric", "count": numeric_count},
            {"label": "Categorical", "count": categorical_count},
            {"label": "Temporal", "count": temporal_count},
            {"label": "Boolean", "count": boolean_count},
        ],
        "metrics": [
            {
                "label": "Telemetry Integrity",
                "value": f"{health_score}/100",
                "hint": health_label,
            },
            {
                "label": "Scan Duration",
                "value": format_elapsed_label(elapsed_ms),
                "hint": "Pipeline completed successfully",
            },
            {
                "label": "Analysis Breadth",
                "value": f"{selection_summary.get('selected_count', 0)} selected",
                "hint": f"{selection_summary.get('recommended_count', 0)} recommended · {selection_summary.get('skipped_count', 0)} skipped",
            },
        ],
        "signals": result.get("signals", []),
        "distribution": distribution,
        "analysisSelection": {
            "selectedAnalyses": analysis_selection.get("selected_analyses", []),
            "recommendedAnalyses": analysis_selection.get("recommended_analyses", []),
            "skippedAnalyses": analysis_selection.get("skipped_analyses", []),
            "summary": selection_summary,
        },
        "health": result.get("health", {}),
        "mlReadiness": result.get("ml_readiness", {}),
        "advanced": result.get("advanced", {}),
        "datasetSignals": result.get("dataset_signals", {}),
        "targetIntelligence": target_intelligence,
        "qualityDiagnostics": quality_diagnostics,
        "deepStatistics": deep_statistics,
        "targetAnalysis": target_analysis,
        "featureImportance": feature_importance,
        "baselineModel": baseline_model,
        "targetLeakage": target_leakage,
        "multicollinearity": multicollinearity,
        "modelDiagnostics": model_diagnostics,
        "segmentAnalysis": segment_analysis,
        "cleaning": result.get("cleaning", {}),
        "charts": result.get("charts", []),
        "aiReport": result.get("ai_report", {}),
        "deepStatisticsV2": deep_statistics_v2,
        "anomaliesV2": anomalies_v2,
        "modelLeaderboard": model_leaderboard,
        "explainability": explainability,
        "timeSeries": time_series_analysis,
        "textProfile": text_profile,
        "timings_ms": result.get("timings_ms", {}),
        "analysisDurationMs": round(float(elapsed_ms), 2),
        "downloadLinks": {
            "cleaned": f"/download-cleaned/{result.get('dataset_id')}",
            "report": f"/download-report/{result.get('dataset_id')}",
        },
    }

    return payload
