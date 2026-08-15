"""
Signal Engine (Display Layer)
==============================
Produces human-readable signal cards for the frontend.
These are DISPLAY signals — the boolean decision flags live in dataset_signals.py.
"""


def severity_from_percent(value):
    if value >= 30:
        return "High"
    if value >= 10:
        return "Medium"
    if value > 0:
        return "Low"
    return "None"


def build_signals(profile, health, ml_readiness, advanced):
    details = health["details"]

    signals = [
        {
            "name": "Data Completeness",
            "icon": "📊",
            "status": "Review" if details["missing_percent"] > 0 else "Good",
            "severity": severity_from_percent(details["missing_percent"]),
            "evidence": f"{details['missing_percent']}% of dataset cells are missing.",
            "recommendation": "Apply suitable imputation or drop columns with excessive missingness.",
        },
        {
            "name": "Duplicate Records",
            "icon": "📋",
            "status": "Review" if profile["duplicate_rows"] > 0 else "Good",
            "severity": severity_from_percent(profile["duplicate_percent"]),
            "evidence": f"{profile['duplicate_rows']} duplicate rows found ({profile['duplicate_percent']}%).",
            "recommendation": "Remove duplicates after verifying they are not valid repeated records.",
        },
        {
            "name": "Outlier Detection",
            "icon": "📈",
            "status": "Review" if details["outlier_percent"] > 0 else "Good",
            "severity": severity_from_percent(details["outlier_percent"]),
            "evidence": f"{details['outlier_percent']}% of numeric cells are potential outliers.",
            "recommendation": "Review outliers before removing or capping. Consider domain context.",
        },
        {
            "name": "Data Consistency",
            "icon": "🔗",
            "status": "Review" if details.get("constant_columns") else "Good",
            "severity": "Medium" if details.get("constant_columns") else "None",
            "evidence": (
                f"{len(details.get('constant_columns', []))} constant column(s) detected."
                if details.get("constant_columns")
                else "No constant columns found."
            ),
            "recommendation": "Remove constant columns — they carry no information.",
        },
        {
            "name": "ML Readiness",
            "icon": "🤖",
            "status": ml_readiness["label"],
            "severity": "High" if ml_readiness["score"] < 50 else "Medium" if ml_readiness["score"] < 75 else "Low",
            "evidence": f"ML readiness score is {ml_readiness['score']}/100.",
            "recommendation": "Handle missing values, encode categoricals, and select a target column.",
        },
        {
            "name": "Anomaly Detection",
            "icon": "🔍",
            "status": "Review" if advanced["anomalies"]["anomalous_rows"] > 0 else "Good",
            "severity": "Medium" if advanced["anomalies"]["anomalous_rows"] > 0 else "None",
            "evidence": f"{advanced['anomalies']['anomalous_rows']} anomalous row(s) detected.",
            "recommendation": "Inspect anomalous rows — they may indicate data entry errors or edge cases.",
        },
        {
            "name": "Fairness Review",
            "icon": "⚖️",
            "status": "Review" if advanced["fairness"]["needs_review"] else "Good",
            "severity": "Informational",
            "evidence": advanced["fairness"]["message"],
            "recommendation": "Check group-wise outcomes before using data for prediction models.",
        },
    ]

    # Add date consistency signal
    if profile.get("date_columns"):
        signals.append({
            "name": "Temporal Data",
            "icon": "📅",
            "status": "Detected",
            "severity": "Informational",
            "evidence": f"{len(profile['date_columns'])} date-like column(s) detected.",
            "recommendation": "Convert date columns to proper datetime format for time-based analysis.",
        })

    return signals
