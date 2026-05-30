"""
Analysis Inventory
==================
Declarative registry of all possible analyses.
Each entry describes: when it runs, what it needs, what it outputs.
The analysis_selector uses this to make data-driven decisions.
"""

ANALYSIS_INVENTORY = [
    {"id": "ingestion_analysis", "name": "Dataset Ingestion Analysis", "category": "Ingestion",
     "modes": ["quick", "standard", "deep", "research"], "priority": "essential",
     "requires": {}, "outputs": ["metadata", "warnings"]},

    {"id": "structural_profile", "name": "Structural Dataset Profiling", "category": "Profiling",
     "modes": ["quick", "standard", "deep", "research"], "priority": "essential",
     "requires": {}, "outputs": ["table", "json", "summary"]},

    {"id": "column_role_analysis", "name": "Column Role Analysis", "category": "Profiling",
     "modes": ["quick", "standard", "deep", "research"], "priority": "essential",
     "requires": {}, "outputs": ["role_map", "summary"]},

    {"id": "missing_value_analysis", "name": "Missing Value Analysis", "category": "Data Quality",
     "modes": ["quick", "standard", "deep", "research"], "priority": "essential",
     "requires": {}, "outputs": ["table", "score", "warning", "chart"]},

    {"id": "high_missingness_analysis", "name": "High Missingness Risk Analysis", "category": "Data Quality",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_high_missingness": True}, "outputs": ["warning", "recommendation", "risk_label"]},

    {"id": "duplicate_analysis", "name": "Duplicate Analysis", "category": "Data Quality",
     "modes": ["quick", "standard", "deep", "research"], "priority": "essential",
     "requires": {}, "outputs": ["count", "warning", "cleaning_action"]},

    {"id": "id_column_detection", "name": "ID Column Detection", "category": "Structural Profiling",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_id_like_columns": True}, "outputs": ["column_list", "warning", "recommendation"]},

    {"id": "descriptive_statistics", "name": "Descriptive Statistics", "category": "Statistics",
     "modes": ["quick", "standard", "deep", "research"], "priority": "essential",
     "requires": {"has_numeric_columns": True}, "outputs": ["table", "json"]},

    {"id": "distribution_analysis", "name": "Distribution Analysis", "category": "Statistics",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_numeric_columns": True}, "outputs": ["chart", "summary"]},

    {"id": "normality_tests", "name": "Normality and Statistical Tests", "category": "Statistics",
     "modes": ["deep", "research"], "priority": "medium",
     "requires": {"has_numeric_columns": True}, "min_rows": 8, "outputs": ["test_result", "p_value"]},

    {"id": "outlier_analysis", "name": "Outlier Analysis", "category": "Statistics",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_numeric_columns": True}, "outputs": ["count", "chart", "warning"]},

    {"id": "correlation_analysis", "name": "Correlation Analysis", "category": "Relationships",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_multiple_numeric_columns": True}, "outputs": ["heatmap", "table", "warning"]},

    {"id": "categorical_analysis", "name": "Categorical Analysis", "category": "Categorical",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_categorical_columns": True}, "outputs": ["frequency_table", "countplot"]},

    {"id": "chi_square_analysis", "name": "Chi-Square Relationship Analysis", "category": "Categorical",
     "modes": ["deep", "research"], "priority": "medium",
     "requires": {"has_categorical_columns": True}, "outputs": ["test_result", "p_value"]},

    {"id": "text_analysis", "name": "Text Column Analysis", "category": "Text",
     "modes": ["deep", "research"], "priority": "medium",
     "requires": {"has_long_text_columns": True}, "outputs": ["text_profile"]},

    {"id": "date_time_analysis", "name": "Date and Time Analysis", "category": "Temporal",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_datetime_columns": True}, "outputs": ["date_range", "freshness"]},

    {"id": "data_quality_scoring", "name": "Data Quality Scoring", "category": "Scoring",
     "modes": ["quick", "standard", "deep", "research"], "priority": "essential",
     "requires": {}, "outputs": ["scorecard", "grade"]},

    {"id": "cleaning_analysis", "name": "Data Cleaning Analysis", "category": "Cleaning",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {}, "outputs": ["cleaning_plan", "cleaned_csv", "audit_log"]},

    {"id": "ml_readiness", "name": "Machine Learning Readiness", "category": "Machine Learning",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {}, "outputs": ["score", "checklist", "recommendation"]},

    {"id": "leakage_detection", "name": "Leakage / Redundancy Detection", "category": "Machine Learning",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {}, "outputs": ["warnings", "severity"]},

    {"id": "target_analysis", "name": "Target Column Analysis", "category": "Machine Learning",
     "modes": ["deep", "research"], "priority": "high",
     "requires": {}, "requires_user_target": True, "outputs": ["target_candidates", "task_type"]},

    {"id": "feature_importance", "name": "Feature Importance Analysis", "category": "Machine Learning",
     "modes": ["deep", "research"], "priority": "high",
     "requires": {"has_numeric_columns": True}, "requires_user_target": True, "outputs": ["ranking", "chart"]},

    {"id": "baseline_model", "name": "Baseline Model Analysis", "category": "Machine Learning",
     "modes": ["deep", "research"], "priority": "high",
     "requires": {"has_numeric_columns": True}, "requires_user_target": True, "outputs": ["metrics"]},

    {"id": "anomaly_detection", "name": "Anomaly Detection", "category": "Anomaly",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires": {"has_numeric_columns": True}, "outputs": ["anomaly_score", "chart"]},

    {"id": "fairness_review", "name": "Bias and Fairness Review", "category": "Responsible AI",
     "modes": ["standard", "deep", "research"], "priority": "medium",
     "requires": {"has_sensitive_column_candidates": True}, "outputs": ["warning"]},

    {"id": "privacy_analysis", "name": "Privacy and Security Analysis", "category": "Privacy",
     "modes": ["standard", "deep", "research"], "priority": "high",
     "requires_any": [{"has_sensitive_column_candidates": True}, {"has_email_like_columns": True}],
     "outputs": ["privacy_score", "pii_warning"]},

    {"id": "time_series_signal", "name": "Time Series Structure Signal", "category": "Temporal",
     "modes": ["deep", "research"], "priority": "medium",
     "requires": {"has_time_series_structure": True}, "outputs": ["warning", "recommendation"]},

    {"id": "ai_summary", "name": "AI Insight Generation", "category": "AI",
     "modes": ["standard", "deep", "research"], "priority": "medium",
     "requires": {}, "outputs": ["natural_language_summary"]},
]
