"""Optional AI interpretation with deterministic local fallbacks.

OpenRouter is attempted first, then Ollama, then a statistics-only fallback.
Environment-backed endpoint metadata is read at call time so test/deployment
configuration is not frozen when the module is imported.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
_DEFAULT_OLLAMA_MODEL = "llama3.2"


def _environment_text(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _openrouter_model() -> str:
    return _environment_text("OPENROUTER_MODEL", _DEFAULT_OPENROUTER_MODEL)


def _ollama_model() -> str:
    return _environment_text("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)


def compact_context(
    profile,
    health,
    signals,
    ml_readiness,
    advanced=None,
    column_roles_summary=None,
    dataset_signals=None,
):
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
        "HTTP-Referer": _environment_text(
            "OPENROUTER_SITE_URL",
            "http://127.0.0.1:5055",
        ),
        "X-Title": _environment_text("OPENROUTER_APP_NAME", "FrameVitals"),
    }


def _call_openrouter(messages, model=None):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": model or _openrouter_model(),
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
        raise RuntimeError(
            f"OpenRouter request failed: {error_payload or exc.reason}"
        ) from exc
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty completion content")
        return content.strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter returned an unexpected payload shape.") from exc


def _call_ollama(messages, model=None):
    try:
        import ollama
    except Exception as exc:
        raise RuntimeError(f"Ollama is unavailable: {exc}") from exc

    response = ollama.chat(
        model=model or _ollama_model(),
        messages=messages,
    )
    try:
        content = response["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty completion content")
        return content.strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Ollama returned an unexpected payload shape.") from exc


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


def _fallback_ai_report(
    profile,
    health,
    ml_readiness,
    advanced,
    roles_summary=None,
    ds_signals=None,
    error=None,
):
    advanced = advanced or {}
    details = health.get("details", {})
    lines = [
        "## Executive Summary",
        (
            f"The dataset contains {profile['shape']['rows']:,} rows and "
            f"{profile['shape']['columns']} columns. The overall health score is "
            f"{health['overall_score']}/100 ({health.get('label', 'Unknown')}). "
            f"ML readiness is {ml_readiness['score']}/100 "
            f"({ml_readiness.get('label', 'Unknown')})."
        ),
        "",
        "## Data Quality Risks",
    ]

    missing_percent = details.get("missing_percent", 0)
    if missing_percent > 0:
        severity = "Critical" if missing_percent >= 30 else "High" if missing_percent >= 10 else "Medium"
        lines.append(f"- **{severity}**: {missing_percent}% of cells are missing")
    if details.get("duplicate_percent", 0) > 0:
        lines.append(
            f"- **Medium**: {details.get('duplicate_percent', 0)}% duplicate rows detected"
        )
    if details.get("outlier_percent", 0) > 0:
        lines.append(
            f"- **Medium**: {details.get('outlier_percent', 0)}% of numeric cells are outliers"
        )
    if details.get("constant_columns"):
        lines.append(
            f"- **Low**: {len(details['constant_columns'])} constant column(s): "
            f"{', '.join(details['constant_columns'])}"
        )
    if details.get("high_cardinality_columns"):
        lines.append(
            f"- **Medium**: {len(details['high_cardinality_columns'])} "
            "high-cardinality column(s) may be identifiers"
        )
    if roles_summary and roles_summary.get("id_like"):
        lines.append(
            f"- **High**: Detected ID-like columns: {', '.join(roles_summary['id_like'])}"
        )
    if ds_signals and ds_signals.get("has_potential_leakage"):
        lines.append("- **Critical**: Potential data leakage detected")
    lines.extend(["", "## Key Insights"])

    if roles_summary and roles_summary.get("target_candidates"):
        lines.append(
            "- Potential target columns: "
            + ", ".join(roles_summary["target_candidates"][:5])
        )
    if roles_summary and roles_summary.get("sensitive"):
        lines.append(
            f"- Sensitive columns detected: {', '.join(roles_summary['sensitive'])}"
        )
    anomalies = advanced.get("anomalies", {})
    if anomalies.get("anomalous_rows", 0) > 0:
        lines.append(
            f"- {anomalies['anomalous_rows']} anomalous rows detected "
            f"(max score: {anomalies.get('highest_score', 'N/A')})"
        )

    lines.extend([
        "",
        "## Cleaning Recommendations",
        "1. Handle missing values before modelling",
    ])
    if details.get("duplicate_percent", 0) > 0:
        lines.append("2. Review and remove duplicate rows")
    lines.append("3. Inspect and handle outliers in numeric columns")
    if roles_summary and roles_summary.get("id_like"):
        lines.append(
            f"4. Remove ID columns before ML: {', '.join(roles_summary['id_like'])}"
        )

    lines.extend([
        "",
        "## ML Readiness Assessment",
        f"Score: {ml_readiness['score']}/100 ({ml_readiness.get('label', 'Unknown')})",
    ])
    for recommendation in ml_readiness.get("recommendations", []):
        lines.append(f"- {recommendation}")

    lines.extend([
        "",
        "## Warnings",
        (
            "- Fairness review: "
            + advanced.get("fairness", {}).get(
                "message",
                "No fairness summary available",
            )
        ),
        (
            "- Leakage status: "
            + advanced.get("leakage", {}).get(
                "status",
                "No leakage summary available",
            )
        ),
        "",
        "## Next Steps",
        "1. Address critical data quality issues first",
        "2. Select a target column for supervised learning",
        "3. Run deep analysis mode for statistical tests",
        "",
        (
            "*Note: This report was generated without a reachable model endpoint. "
            "It is based on computed statistics and heuristics.*"
        ),
    ])

    source = "fallback" if not error else f"fallback: {error}"
    return {"source": source, "text": "\n".join(lines)}


def _fallback_answer(profile, health, error=None):
    source = "fallback" if not error else f"fallback: {error}"
    return {
        "source": source,
        "answer": (
            "## Answer\n"
            "A model endpoint is not currently reachable.\n\n"
            "## Evidence\n"
            f"The dataset has {profile['shape']['rows']:,} rows, "
            f"{profile['shape']['columns']} columns, and a health score of "
            f"{health['overall_score']}/100.\n\n"
            "## Recommendation\n"
            "Start by reviewing missing values, duplicates, outliers, and "
            "ML-readiness indicators.\n\n"
            "## Limitation\n"
            "This answer is generated without a reachable model endpoint."
        ),
    }


def generate_ai_report(
    profile,
    health,
    signals,
    ml_readiness,
    advanced,
    column_roles_summary=None,
    dataset_signals=None,
    model=None,
):
    context = compact_context(
        profile,
        health,
        signals,
        ml_readiness,
        advanced,
        column_roles_summary,
        dataset_signals,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert data scientist writing a professional dataset "
                "analysis report. Be evidence-based, concise, and actionable."
            ),
        },
        {"role": "user", "content": _build_report_prompt(context)},
    ]

    try:
        text = _call_openrouter(messages, model=model)
        return {"source": "openrouter", "text": text}
    except Exception as openrouter_error:
        try:
            text = _call_ollama(messages, model=model or _ollama_model())
            return {"source": "ollama", "text": text}
        except Exception as ollama_error:
            return _fallback_ai_report(
                profile,
                health,
                ml_readiness,
                advanced,
                column_roles_summary,
                dataset_signals,
                error=f"{openrouter_error}; {ollama_error}",
            )


def answer_dataset_question(
    question,
    profile,
    health,
    signals,
    ml_readiness,
    advanced=None,
    model=None,
):
    context = compact_context(profile, health, signals, ml_readiness, advanced)
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from the provided dataset context. "
                "Be precise and evidence-based."
            ),
        },
        {"role": "user", "content": _build_question_prompt(question, context)},
    ]

    try:
        text = _call_openrouter(messages, model=model)
        return {"source": "openrouter", "answer": text}
    except Exception as openrouter_error:
        try:
            text = _call_ollama(messages, model=model or _ollama_model())
            return {"source": "ollama", "answer": text}
        except Exception as ollama_error:
            return _fallback_answer(
                profile,
                health,
                error=f"{openrouter_error}; {ollama_error}",
            )
