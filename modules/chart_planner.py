"""
Chart Planner
=============
Decides WHAT charts to generate based on column roles and signals.
Avoids meaningless charts (e.g., histograms of ID columns).

Receives optional `context` dict with the upstream analytical phases so the
planner can light up advanced charts (leaderboard bars, target overlays,
time-series trends, etc.) when those phases produced data.
"""

from modules.column_roles import (
    get_meaningful_numeric_columns,
    get_meaningful_categorical_columns,
    get_columns_with_role,
)


def build_chart_plan(
    df,
    profile,
    health,
    advanced,
    cleaning,
    column_roles,
    context: dict | None = None,
):
    context = context or {}
    target_column = context.get("target_column")
    model_leaderboard = context.get("model_leaderboard") or {}
    explainability = context.get("explainability") or {}
    time_series = context.get("time_series") or {}
    deep_stats_v2 = context.get("deep_statistics_v2") or {}

    numeric = get_meaningful_numeric_columns(df, column_roles)
    categorical = get_meaningful_categorical_columns(df, column_roles)
    price_cols = get_columns_with_role(column_roles, "price_like")
    volume_cols = get_columns_with_role(column_roles, "volume_like")
    high_miss = [c for c, info in column_roles.items() if info["missing_percent"] >= 20]

    plan: list[dict] = []

    # ===================================================================
    # Quality / structure
    # ===================================================================

    plan.append({"id": "health", "type": "health_components",
                 "title": "Dataset Health Components",
                 "reason": "Always useful for quality overview."})

    # NEW: dtype breakdown donut — visual at-a-glance schema makeup
    plan.append({"id": "dtype_breakdown", "type": "dtype_breakdown",
                 "title": "Schema Composition",
                 "reason": "Visual breakdown of column dtypes."})

    if profile.get("missing_counts") and sum(
        v for v in profile["missing_counts"].values() if isinstance(v, (int, float))
    ) > 0:
        plan.append({"id": "missing", "type": "missing_values",
                     "title": "Missing Values by Column",
                     "reason": "Dataset has missing values."})

    if high_miss:
        plan.append({"id": "cleaning_risk", "type": "cleaning_risk",
                     "title": "Cleaning Risk by Column",
                     "reason": "Columns with ≥20% missing detected."})

    # NEW: cardinality strip — surfaces ID-like / constant / high-cardinality cats
    plan.append({"id": "cardinality", "type": "cardinality_strip",
                 "title": "Column Cardinality",
                 "reason": "Quick scan for ID-like, constant, or high-cardinality columns."})

    # ===================================================================
    # Distributions (numeric)
    # ===================================================================

    dist_col = None
    for c in price_cols:
        if c in numeric:
            dist_col = c
            break
    if not dist_col and numeric:
        dist_col = numeric[0]

    if dist_col:
        plan.append({"id": f"dist_{dist_col}", "type": "numeric_distribution",
                     "column": dist_col, "title": f"Distribution of {dist_col}",
                     "reason": "Meaningful numeric column selected."})
        plan.append({"id": f"box_{dist_col}", "type": "boxplot",
                     "column": dist_col, "title": f"Outlier Review: {dist_col}",
                     "reason": "Outlier inspection for selected column."})
        # NEW: violin gives shape + outliers in one
        plan.append({"id": f"violin_{dist_col}", "type": "violin",
                     "column": dist_col, "title": f"Density Violin: {dist_col}",
                     "reason": "Shape + density of the primary numeric column."})

    # Second + third numeric distribution if available (was: only 2nd)
    extras = [c for c in numeric if c != dist_col][:2]
    for second in extras:
        plan.append({"id": f"dist_{second}", "type": "numeric_distribution",
                     "column": second, "title": f"Distribution of {second}",
                     "reason": "Additional meaningful numeric column."})

    # NEW: numeric distribution strip (one figure showing the top 6 numerics)
    if len(numeric) >= 3:
        plan.append({"id": "numeric_overview", "type": "numeric_overview",
                     "columns": numeric[:6],
                     "title": "Numeric Columns at a Glance",
                     "reason": "Compact overview of multiple numeric distributions."})

    lower_map = {c.lower(): c for c in df.columns}
    if "bid" in lower_map and "ask" in lower_map:
        plan.append({"id": "bid_ask", "type": "bid_ask_spread",
                     "bid_column": lower_map["bid"], "ask_column": lower_map["ask"],
                     "title": "Bid-Ask Spread", "reason": "Bid/ask columns detected."})

    for c in volume_cols:
        if c in numeric and c != dist_col:
            plan.append({"id": f"vol_{c}", "type": "numeric_distribution",
                         "column": c, "title": f"Distribution of {c}",
                         "reason": "Volume-like column detected."})
            break

    # ===================================================================
    # Categoricals
    # ===================================================================

    for c in categorical[:2]:
        plan.append({"id": f"cat_{c}", "type": "categorical_count",
                     "column": c, "title": f"Categories in {c}",
                     "reason": "Meaningful categorical column."})

    # NEW: pareto chart on the busiest categorical (cumulative coverage)
    if categorical:
        plan.append({"id": f"pareto_{categorical[0]}", "type": "pareto_categorical",
                     "column": categorical[0],
                     "title": f"Pareto: {categorical[0]}",
                     "reason": "Cumulative share of top categories."})

    # ===================================================================
    # Relationships
    # ===================================================================

    if len(profile.get("numeric_columns", [])) >= 2:
        plan.append({"id": "corr", "type": "correlation_heatmap",
                     "title": "Correlation Heatmap",
                     "reason": "Multiple numeric columns available."})
        # NEW: top correlation pairs as a bar chart (more readable than the heatmap alone)
        plan.append({"id": "top_corr", "type": "top_correlations",
                     "title": "Top Correlations",
                     "reason": "Strongest pairwise relationships at a glance."})

    if advanced.get("column_utility"):
        plan.append({"id": "utility", "type": "column_utility",
                     "title": "Column Utility Scores",
                     "reason": "Ranks columns by analytical usefulness."})

    plan.append({"id": "cleaning_impact", "type": "cleaning_impact",
                 "title": "Cleaning Impact",
                 "reason": "Before vs after quality comparison."})

    if advanced.get("anomalies", {}).get("top_rows"):
        plan.append({"id": "anomaly", "type": "anomaly_scores",
                     "title": "Top Anomaly Scores",
                     "reason": "Anomalous rows detected."})

    # ===================================================================
    # Target-aware
    # ===================================================================

    if target_column and target_column in df.columns:
        plan.append({"id": "target_dist", "type": "target_distribution",
                     "column": target_column,
                     "title": f"Target Distribution: {target_column}",
                     "reason": "Show class balance / regression target shape."})

        # Target vs the top numeric (excluding the target itself)
        for c in numeric[:3]:
            if c == target_column:
                continue
            plan.append({
                "id": f"tvf_{c}",
                "type": "target_vs_feature",
                "feature": c,
                "target": target_column,
                "title": f"{c} vs {target_column}",
                "reason": "Top numeric feature against the target.",
            })
            break

    # ===================================================================
    # Modeling
    # ===================================================================

    if model_leaderboard.get("available"):
        rows = model_leaderboard.get("leaderboard") or []
        scored = [r for r in rows if r.get("primary_score") is not None]
        if scored:
            plan.append({"id": "leaderboard", "type": "leaderboard_bars",
                         "title": "Model Leaderboard",
                         "reason": "Cross-validated model scores."})

    if explainability.get("available") and explainability.get("global_importance"):
        plan.append({"id": "feature_importance", "type": "feature_importance_bars",
                     "title": "Feature Importance",
                     "reason": "Mean |SHAP| from the leaderboard winner."})

    # ===================================================================
    # Time series
    # ===================================================================

    if time_series.get("available"):
        plan.append({"id": "ts_trend", "type": "time_series_trend",
                     "title": "Time-series Trend",
                     "reason": "Detected date column with time-series structure."})

    # ===================================================================
    # Bivariate / deep stats highlights
    # ===================================================================

    if deep_stats_v2:
        biv = (deep_stats_v2.get("bivariate") or {})
        if (biv.get("numeric_pairs") or biv.get("categorical_pairs") or biv.get("group_difference_tests")):
            plan.append({"id": "bivariate_highlights", "type": "bivariate_highlights",
                         "title": "Bivariate Highlights",
                         "reason": "Strongest numeric / categorical relationships from deep statistics."})

    return plan
