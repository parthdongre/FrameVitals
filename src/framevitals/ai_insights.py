"""
AI Insights
===========
OpenRouter-first AI report generation with Ollama fallback and a deterministic
rule-based fallback when no model is reachable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct:free",
)
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OPENROUTER_SITE_URL = os.environ.get("OPENROUTER_SITE_URL", "http://127.0.0.1:5055")
OPENROUTER_APP_NAME = os.environ.get("OPENROUTER_APP_NAME", "DataLens AI")


def compact_context(profile, health, signals, ml_readiness, advanced=None, column_roles_summary=None, dataset_signals=None):
    context = {
        "profile": {
            "shape": profile["shape"],
            "columns": profile["columns"],
            "dtypes": profile["dtypes"],
            "missing_counts": profile["missing_counts"],
            "duplicate_rows": profile["duplicate_rows"],
            "numeric_columns": profile["numeric_columns"],
            "categorical_columns": profile["categorical_columns"],
            "date_columns": profile["date_columns"],
        },
        "health": health,
        "signals": signals,
        "ml_readiness": ml_readiness,
    }

    if advanced:
        context["advanced"] = {
            "anomalies": advanced.get("anomalies", {}),
            "fairness": advanced.get("fairness", {}),
            "freshness": advanced.get("freshness", {}),
            "leakage": advanced.get("leakage", {}),
            "top_column_utility": advanced.get("column_utility", [])[:8],
        }

    if column_roles_summary:
        context["column_roles_summary"] = column_roles_summary

    if dataset_signals:
        context["dataset_signals"] = dataset_signals

    return context


def _openrouter_headers():
    return {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '').strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }


def _call_openrouter(messages, model=None):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": model or DEFAULT_OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1400,
    }

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=_openrouter_headers(),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_payload = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenRouter request failed: {error_payload or exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"OpenRouter returned an unexpected payload: {data}") from exc


def _call_ollama(messages, model=None):
    try:
        import ollama
    except Exception as exc:
        raise RuntimeError(f"Ollama is unavailable: {exc}") from exc

    response = ollama.chat(
        model=model or DEFAULT_OLLAMA_MODEL,
        messages=messages,
    )
    return response["message"]["content"].strip()


def _build_report_prompt(context):
    return f"""Analyze this dataset summary and produce a structured report.

Dataset Context:
{json.dumps(context, indent=2)}

Required output format:

## Executive Summary

## Data Quality Risks

## Key Insights

## Cleaning Recommendations

## ML Readiness Assessment

## Warnings

## Next Steps"""


def _build_question_prompt(question, context):
    return f"""Answer the user's question using only the dataset context below.
Be specific, cite evidence, and mention limitations.

Context:
{json.dumps(context, indent=2)}

Question: {question}

Format your answer as:
## Answer
## Evidence
## Recommendation
## Limitation"""


def _fallback_ai_report(profile, health, ml_readiness, advanced, roles_summary=None, ds_signals=None, error=None):
    details = health.get("details", {})
    lines = []

    lines.append("## Executive Summary")
    lines.append(
        f"The dataset contains {profile['shape']['rows']:,} rows and {profile['shape']['columns']} columns. "
        f"The overall health score is {health['overall_score']}/100 ({health.get('label', 'Unknown')}). "
        f"ML readiness is {ml_readiness['score']}/100 ({ml_readiness.get('label', 'Unknown')})."
    )
    lines.append("")

    lines.append("## Data Quality Risks")
    if details.get("missing_percent", 0) > 0:
        sev = "Critical" if details.get("missing_percent", 0) >= 30 else "High" if details.get("missing_percent", 0) >= 10 else "Medium"
        lines.append(f"- **{sev}**: {details.get('missing_percent', 0)}% of cells are missing")
    if details.get("duplicate_percent", 0) > 0:
        lines.append(f"- **Medium**: {details.get('duplicate_percent', 0)}% duplicate rows detected")
    if details.get("outlier_percent", 0) > 0:
        lines.append(f"- **Medium**: {details.get('outlier_percent', 0)}% of numeric cells are outliers")
    if details.get("constant_columns"):
        lines.append(f"- **Low**: {len(details['constant_columns'])} constant column(s): {', '.join(details['constant_columns'])}")
    if details.get("high_cardinality_columns"):
        lines.append(f"- **Medium**: {len(details['high_cardinality_columns'])} high-cardinality column(s) may be identifiers")
    if roles_summary and roles_summary.get("id_like"):
        lines.append(f"- **High**: Detected ID-like columns: {', '.join(roles_summary['id_like'])}")
    if ds_signals and ds_signals.get("has_potential_leakage"):
        lines.append("- **Critical**: Potential data leakage detected")
    lines.append("")

    lines.append("## Key Insights")
    if roles_summary and roles_summary.get("target_candidates"):
        lines.append(f"- Potential target columns: {', '.join(roles_summary['target_candidates'][:5])}")
    if roles_summary and roles_summary.get("sensitive"):
        lines.append(f"- Sensitive columns detected: {', '.join(roles_summary['sensitive'])}")
    anomalies = advanced.get("anomalies", {})
    if anomalies.get("anomalous_rows", 0) > 0:
        lines.append(f"- {anomalies['anomalous_rows']} anomalous rows detected (max score: {anomalies.get('highest_score', 'N/A')})")
    lines.append("")

    lines.append("## Cleaning Recommendations")
    lines.append("1. Handle missing values before modelling")
    if details.get("duplicate_percent", 0) > 0:
        lines.append("2. Review and remove duplicate rows")
    lines.append("3. Inspect and handle outliers in numeric columns")
    if roles_summary and roles_summary.get("id_like"):
        lines.append(f"4. Remove ID columns before ML: {', '.join(roles_summary['id_like'])}")
    lines.append("")

    lines.append("## ML Readiness Assessment")
    lines.append(f"Score: {ml_readiness['score']}/100 ({ml_readiness.get('label', 'Unknown')})")
    for recommendation in ml_readiness.get("recommendations", []):
        lines.append(f"- {recommendation}")
    lines.append("")

    lines.append("## Warnings")
    lines.append(f"- Fairness review: {advanced.get('fairness', {}).get('message', 'No fairness summary available')}")
    lines.append(f"- Leakage status: {advanced.get('leakage', {}).get('status', 'No leakage summary available')}")
    lines.append("")

    lines.append("## Next Steps")
    lines.append("1. Address critical data quality issues first")
    lines.append("2. Select a target column for supervised learning")
    lines.append("3. Run deep analysis mode for statistical tests")
    lines.append("")
    lines.append(
        "*Note: This report was generated without a reachable model endpoint. It is based on computed statistics and heuristics.*"
    )

    source = "fallback"
    if error:
        source = f"fallback: {error}"

    return {"source": source, "text": "\n".join(lines)}


def _fallback_answer(profile, health, error=None):
    source = "fallback"
    if error:
        source = f"fallback: {error}"

    return {
        "source": source,
        "answer": (
            "## Answer\n"
            "A model endpoint is not currently reachable.\n\n"
            "## Evidence\n"
            f"The dataset has {profile['shape']['rows']:,} rows, {profile['shape']['columns']} columns, and a health score of {health['overall_score']}/100.\n\n"
            "## Recommendation\n"
            "Start by reviewing missing values, duplicates, outliers, and ML-readiness indicators.\n\n"
            "## Limitation\n"
            "This answer is generated without a reachable model endpoint."
        ),
    }


def generate_ai_report(profile, health, signals, ml_readiness, advanced, column_roles_summary=None, dataset_signals=None, model=None):
    context = compact_context(profile, health, signals, ml_readiness, advanced, column_roles_summary, dataset_signals)
    prompt = _build_report_prompt(context)
    messages = [
        {
            "role": "system",
            "content": "You are an expert data scientist writing a professional dataset analysis report. Be evidence-based, concise, and actionable.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        text = _call_openrouter(messages, model=model)
        return {"source": "openrouter", "text": text}
    except Exception as openrouter_error:
        try:
            text = _call_ollama(messages, model=model or DEFAULT_OLLAMA_MODEL)
            return {"source": "ollama", "text": text}
        except Exception as ollama_error:
            return {
                "source": f"fallback: {openrouter_error}; {ollama_error}",
                **_fallback_ai_report(profile, health, ml_readiness, advanced, column_roles_summary, dataset_signals, None),
            }


def answer_dataset_question(question, profile, health, signals, ml_readiness, advanced=None, model=None):
    context = compact_context(profile, health, signals, ml_readiness, advanced)
    prompt = _build_question_prompt(question, context)
    messages = [
        {
            "role": "system",
            "content": "Answer only from the provided dataset context. Be precise and evidence-based.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        text = _call_openrouter(messages, model=model)
        return {"source": "openrouter", "answer": text}
    except Exception as openrouter_error:
        try:
            text = _call_ollama(messages, model=model or DEFAULT_OLLAMA_MODEL)
            return {"source": "ollama", "answer": text}
        except Exception as ollama_error:
            return {
                "source": f"fallback: {openrouter_error}; {ollama_error}",
                **_fallback_answer(profile, health, None),
            }
