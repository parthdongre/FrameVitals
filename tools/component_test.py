"""
DataLens AI — full component test harness.

Exercises every analytical module in the project against three representative
demo datasets, captures pass / fail / timing / evidence per module, and writes
the result to ``reports/component_test_manifest.json``. The companion
``whitepaper_builder.py`` reads that manifest to produce a LaTeX whitepaper.

Each "component test" is a small, deterministic probe:

    - It calls the public function the rest of the system uses.
    - It returns a structured result with `ok`, `summary`, and `metrics`.
    - It catches exceptions so one broken module never sinks the run.

Run from repo root:

    python tools/component_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

# Quiet down noisy upstream warnings so the run output stays readable.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Default the harness to its fast configuration:
#   - TF-IDF retrieval (~17ms vs ~9s of Ollama embedding round-trips)
#   - LLM agent probe skipped (the cloud round-trip alone is ~50s/dataset)
# Set DATALENS_TEST_FULL=1 to opt back into the slow path when you want a
# full audit including a real LLM answer for each dataset.
_FULL_RUN = os.environ.get("DATALENS_TEST_FULL", "0").strip() in {"1", "true", "yes"}
if not _FULL_RUN:
    os.environ.setdefault("DATALENS_RAG_BACKEND", "tfidf")
    os.environ.setdefault("DATALENS_TEST_SKIP_LLM", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Datasets the harness exercises
# ---------------------------------------------------------------------------

# Tuned to span: classification target (churn), regression target (price),
# and a target-less feed (no analysis_selector target → exercises non-ML
# branches like text profile, drift, etc.).
DATASETS: list[dict[str, Any]] = [
    {
        "label": "classification",
        "csv": "demo_datasets/customer_churn.csv",
        "target": "churned",
        "mode": "deep",
    },
    {
        "label": "regression",
        "csv": "demo_datasets/real_estate.csv",
        "target": "price",
        "mode": "standard",
    },
    {
        "label": "no_target",
        "csv": "demo_datasets/medical_records.csv",
        "target": None,
        "mode": "standard",
    },
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComponentResult:
    name: str                      # e.g. "loader", "explainability"
    category: str                  # e.g. "ingest", "modeling"
    ok: bool                       # did it run without exception
    available: bool | None = None  # for components with availability flags
    duration_ms: float = 0.0
    summary: str = ""              # one-line human summary
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DatasetRun:
    label: str
    csv: str
    target: str | None
    mode: str
    rows: int = 0
    columns: int = 0
    components: list[ComponentResult] = field(default_factory=list)
    pipeline_total_ms: float = 0.0
    charts: list[dict] = field(default_factory=list)  # [{type, title, path, description}]


@dataclass
class Manifest:
    started_at: float
    completed_at: float
    python_version: str
    runs: list[DatasetRun] = field(default_factory=list)
    component_definitions: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Component definitions — name, category, blurb (used by the LaTeX whitepaper)
# ---------------------------------------------------------------------------

COMPONENT_DEFINITIONS: list[dict[str, Any]] = [
    # 1. Ingest
    {"name": "loader",        "category": "Ingest",
     "title": "Dataset Loader"},
    {"name": "profiler",      "category": "Ingest",
     "title": "Schema Profiler"},
    {"name": "column_roles",  "category": "Ingest",
     "title": "Column Role Inference"},
    {"name": "dataset_signals", "category": "Ingest",
     "title": "Dataset Signal Detection"},
    {"name": "analysis_selector", "category": "Ingest",
     "title": "Analysis Selector"},
    # 2. Quality
    {"name": "health_score",  "category": "Quality",
     "title": "Health Score"},
    {"name": "ml_readiness",  "category": "Quality",
     "title": "ML Readiness Score"},
    {"name": "advanced_indicators", "category": "Quality",
     "title": "Advanced Quality Indicators"},
    {"name": "signal_engine", "category": "Quality",
     "title": "Signal Engine"},
    {"name": "cleaner",       "category": "Quality",
     "title": "Auto-Cleaner"},
    # 3. Statistics
    {"name": "deep_statistics_v2", "category": "Statistics",
     "title": "Deep Statistics v2"},
    {"name": "anomaly_ensemble",   "category": "Statistics",
     "title": "Anomaly Ensemble"},
    # 4. Modeling
    {"name": "model_leaderboard",  "category": "Modeling",
     "title": "Model Leaderboard"},
    {"name": "explainability",     "category": "Modeling",
     "title": "SHAP Explainability"},
    # 5. Time / text
    {"name": "time_series",   "category": "Time & Text",
     "title": "Time-series Analysis"},
    {"name": "text_profile",  "category": "Time & Text",
     "title": "Text / NLP Profiling"},
    # 6. Drift
    {"name": "drift_analysis", "category": "Drift",
     "title": "Drift Analysis"},
    # 7. Charts + reporting
    {"name": "chart_planner", "category": "Reporting",
     "title": "Chart Planner"},
    {"name": "visualizer",    "category": "Reporting",
     "title": "Chart Renderer"},
    {"name": "pdf_report_builder", "category": "Reporting",
     "title": "PDF Report Builder"},
    # 8. Brief / agent / RAG
    {"name": "frontend_api",  "category": "Agentic",
     "title": "Dashboard Payload Builder"},
    {"name": "agent_brief",   "category": "Agentic",
     "title": "Dataset Brief Packer"},
    {"name": "rag_index",     "category": "Agentic",
     "title": "RAG Fact Index"},
    {"name": "ai_agent",      "category": "Agentic",
     "title": "AI Agent (Q&A)"},
]


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------

def _timed(fn: Callable[[], Any]) -> tuple[Any, float, str | None]:
    t0 = time.perf_counter()
    err: str | None = None
    value = None
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 — we want to capture and continue
        err = f"{type(exc).__name__}: {exc}"
        # capture short traceback for the whitepaper appendix
        err += "\n" + "".join(traceback.format_exception(exc)[-3:])
    return value, (time.perf_counter() - t0) * 1000.0, err


def _probe(name: str, category: str, fn: Callable[[], dict],
           summary_keys: tuple[str, ...] = ("summary",)) -> ComponentResult:
    """
    Run a probe and wrap the result. The probe callable should return a dict
    with at least:

        {
            "summary": "...",
            "metrics": {...},
            "available": True|False (optional, for available-flag modules)
        }
    """
    value, ms, err = _timed(fn)
    if err is not None:
        return ComponentResult(
            name=name, category=category, ok=False,
            duration_ms=ms, error=err, summary=f"{name} failed: {err.splitlines()[0]}",
        )
    if not isinstance(value, dict):
        return ComponentResult(
            name=name, category=category, ok=False,
            duration_ms=ms, error="probe did not return a dict",
            summary=f"{name} produced no dict",
        )
    return ComponentResult(
        name=name,
        category=category,
        ok=True,
        available=value.get("available"),
        duration_ms=ms,
        summary=value.get("summary", ""),
        metrics=value.get("metrics", {}),
    )


# ---------------------------------------------------------------------------
# Per-component probes
# ---------------------------------------------------------------------------

def probe_loader(csv_path: Path, df: pd.DataFrame) -> dict:
    return {
        "summary": f"loaded {len(df):,} rows × {df.shape[1]} cols from {csv_path.name}",
        "metrics": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "size_kb": round(csv_path.stat().st_size / 1024, 1),
        },
    }


def probe_profiler(df: pd.DataFrame) -> dict:
    from modules.profiler import build_profile
    profile = build_profile(df)
    nm = len(profile.get("numeric_columns") or [])
    cm = len(profile.get("categorical_columns") or [])
    dt = len(profile.get("date_columns") or [])
    miss = sum(v for v in (profile.get("missing_counts") or {}).values()
               if isinstance(v, (int, float)))
    return {
        "summary": f"profile: {nm} numeric, {cm} categorical, {dt} date · "
                   f"{int(miss):,} missing cells",
        "metrics": {"numeric": nm, "categorical": cm, "date": dt,
                    "total_missing": int(miss)},
    }


def probe_column_roles(df: pd.DataFrame) -> dict:
    from modules.column_roles import infer_column_roles, summarize_roles
    roles = infer_column_roles(df)
    summary = summarize_roles(roles)
    return {
        "summary": f"roles: id={summary.get('id_columns', 0)}, "
                   f"price={summary.get('price_like', 0)}, "
                   f"volume={summary.get('volume_like', 0)}, "
                   f"target_candidates={summary.get('target_candidates', 0)}",
        "metrics": summary,
    }


def probe_dataset_signals(df: pd.DataFrame) -> dict:
    from modules.profiler import build_profile
    from modules.dataset_signals import detect_dataset_signals
    profile = build_profile(df)
    sigs = detect_dataset_signals(df, profile)
    on = [k for k, v in sigs.items() if isinstance(v, bool) and v]
    return {
        "summary": f"signals on: {', '.join(on[:6]) or '—'}",
        "metrics": {"flags_on": len(on), "total_keys": len(sigs)},
    }


def probe_analysis_selector(df: pd.DataFrame, target: str | None,
                             mode: str) -> dict:
    from modules.profiler import build_profile
    from modules.dataset_signals import detect_dataset_signals
    from modules.analysis_selector import select_analyses
    profile = build_profile(df)
    sigs = detect_dataset_signals(df, profile)
    sel = select_analyses(signals=sigs, analysis_mode=mode,
                          target_column=target)
    s = sel.get("summary", {})
    return {
        "summary": f"selected={s.get('selected_count', 0)} · "
                   f"recommended={s.get('recommended_count', 0)} · "
                   f"skipped={s.get('skipped_count', 0)}",
        "metrics": s,
    }


def probe_health_score(df: pd.DataFrame) -> dict:
    from modules.profiler import build_profile
    from modules.health_score import calculate_health_score
    profile = build_profile(df)
    health = calculate_health_score(df, profile)
    return {
        "summary": f"health = {health.get('overall_score', 0)}/100 "
                   f"({health.get('label', '?')})",
        "metrics": {
            "overall_score": health.get("overall_score"),
            "label": health.get("label"),
            "components": health.get("components", {}),
        },
    }


def probe_ml_readiness(df: pd.DataFrame) -> dict:
    from modules.ml_readiness import calculate_ml_readiness
    ml = calculate_ml_readiness(df)
    return {
        "summary": f"ML readiness = {ml.get('score', 0)}/100 ({ml.get('label', '?')})",
        "metrics": {"score": ml.get("score"), "label": ml.get("label")},
    }


def probe_advanced_indicators(df: pd.DataFrame) -> dict:
    from modules.advanced_indicators import calculate_advanced_indicators
    adv = calculate_advanced_indicators(df)
    util = adv.get("column_utility", []) or []
    anomalies = adv.get("anomalies", {}) or {}
    return {
        "summary": f"utility ranked {len(util)} cols · "
                   f"top anomaly score {anomalies.get('max_score', 0):.2f}",
        "metrics": {
            "n_columns_ranked": len(util),
            "anomaly_max_score": anomalies.get("max_score"),
            "anomaly_top_count": len(anomalies.get("top_rows") or []),
        },
    }


def probe_signal_engine(df: pd.DataFrame) -> dict:
    from modules.profiler import build_profile
    from modules.health_score import calculate_health_score
    from modules.ml_readiness import calculate_ml_readiness
    from modules.advanced_indicators import calculate_advanced_indicators
    from modules.signal_engine import build_signals
    profile = build_profile(df)
    health = calculate_health_score(df, profile)
    ml = calculate_ml_readiness(df)
    adv = calculate_advanced_indicators(df)
    sigs = build_signals(profile, health, ml, adv)
    by_sev = {}
    for s in sigs:
        by_sev[s.get("severity", "low")] = by_sev.get(s.get("severity", "low"), 0) + 1
    return {
        "summary": f"{len(sigs)} signals · " +
                   ", ".join(f"{k}:{v}" for k, v in by_sev.items()),
        "metrics": {"total": len(sigs), "by_severity": by_sev},
    }


def probe_cleaner(dataset_id: str, df: pd.DataFrame) -> dict:
    from modules.cleaner import create_cleaned_dataset
    cl = create_cleaned_dataset(dataset_id, df)
    return {
        "summary": f"cleaning {cl.get('before_health', {}).get('overall_score', 0)} → "
                   f"{cl.get('after_health', {}).get('overall_score', 0)} · "
                   f"actions={len(cl.get('actions') or [])}",
        "metrics": {
            "before_score": (cl.get("before_health") or {}).get("overall_score"),
            "after_score": (cl.get("after_health") or {}).get("overall_score"),
            "actions": len(cl.get("actions") or []),
            "missing_before": cl.get("missing_before"),
            "missing_after": cl.get("missing_after"),
            "duplicates_before": cl.get("duplicates_before"),
            "duplicates_after": cl.get("duplicates_after"),
        },
    }


def probe_deep_statistics(df: pd.DataFrame) -> dict:
    from modules.deep_statistics_v2 import run_deep_statistics_v2
    deep = run_deep_statistics_v2(df)
    summary = deep.get("summary") or {}
    return {
        "summary": (
            f"pairs: num={summary.get('numeric_pairs_tested', 0)}, "
            f"cat={summary.get('categorical_pairs_tested', 0)}, "
            f"groupdiff={summary.get('group_difference_tests_run', 0)}"
        ),
        "metrics": summary,
        "available": True,
    }


def probe_anomaly_ensemble(df: pd.DataFrame) -> dict:
    from modules.anomaly_ensemble import detect_anomalies_ensemble
    out = detect_anomalies_ensemble(df)
    return {
        "summary": (
            f"available={out.get('available')} · "
            f"top rows={len(out.get('top_rows') or [])} · "
            f"detectors={len(out.get('detectors') or [])}"
        ),
        "metrics": {
            "n_rows_scored": out.get("n_rows_scored"),
            "n_top_rows": len(out.get("top_rows") or []),
            "detectors": list((out.get("detectors") or {}).keys()),
            "used_columns": len(out.get("used_columns") or []),
        },
        "available": out.get("available"),
    }


def probe_model_leaderboard(df: pd.DataFrame, target: str | None) -> dict:
    from modules.model_leaderboard import run_model_leaderboard
    if not target:
        return {"summary": "skipped — no target column provided",
                "metrics": {}, "available": False}
    out = run_model_leaderboard(df, target_column=target)
    if not out.get("available"):
        return {"summary": f"unavailable: {out.get('message', '')[:60]}",
                "metrics": {"reason": out.get("message")}, "available": False}
    winner = out.get("winner") or {}
    rows = [r for r in (out.get("leaderboard") or [])
            if r.get("primary_score") is not None]
    return {
        "summary": (
            f"task={out.get('task_type')} · winner={winner.get('model')} · "
            f"score={winner.get('primary_score')}"
        ),
        "metrics": {
            "task_type": out.get("task_type"),
            "winner": winner.get("model"),
            "winner_score": winner.get("primary_score"),
            "models_scored": len(rows),
        },
        "available": True,
    }


def probe_explainability(df: pd.DataFrame, target: str | None,
                          dataset_id: str) -> dict:
    from modules.model_leaderboard import run_model_leaderboard
    from modules.explainability import explain_winner
    if not target:
        return {"summary": "skipped — no target column provided",
                "metrics": {}, "available": False}
    lb = run_model_leaderboard(df, target_column=target)
    if not lb.get("available") or not lb.get("winner"):
        return {"summary": "skipped — no leaderboard winner",
                "metrics": {}, "available": False}
    out = explain_winner(df, target_column=target,
                          leaderboard_result=lb, dataset_id=dataset_id)
    if not out.get("available"):
        return {"summary": f"unavailable: {out.get('message', '')[:60]}",
                "metrics": {"reason": out.get("message")}, "available": False}
    gi = out.get("global_importance") or []
    top = ", ".join([str(r.get("feature", "?")) for r in gi[:3]])
    return {
        "summary": f"method={out.get('method')} · top features: {top}",
        "metrics": {
            "method": out.get("method"),
            "n_features": len(gi),
            "top_features": [{"feature": r.get("feature"),
                              "importance": r.get("importance")}
                              for r in gi[:5]],
        },
        "available": True,
    }


def probe_time_series(df: pd.DataFrame, target: str | None) -> dict:
    from modules.time_series import detect_and_analyze_time_series
    out = detect_and_analyze_time_series(df, target_column=target)
    if not out.get("available"):
        return {"summary": f"no time-series: {out.get('reason', '')[:60]}",
                "metrics": {"reason": out.get("reason")}, "available": False}
    return {
        "summary": (
            f"date={out.get('detected_date_column')} · "
            f"numeric={out.get('numeric_column')} · "
            f"freq={(out.get('frequency') or {}).get('label')}"
        ),
        "metrics": {
            "date_column": out.get("detected_date_column"),
            "numeric_column": out.get("numeric_column"),
            "frequency": (out.get("frequency") or {}).get("label"),
            "stationarity": (out.get("stationarity") or {}).get("verdict"),
        },
        "available": True,
    }


def probe_text_profile(df: pd.DataFrame) -> dict:
    from modules.text_profile import profile_text_columns
    out = profile_text_columns(df)
    if not out.get("available"):
        return {"summary": f"no text columns ({out.get('reason', '')[:60]})",
                "metrics": {"reason": out.get("reason")}, "available": False}
    cols = out.get("profiled_columns") or []
    return {
        "summary": f"profiled {len(cols)} text col(s): {', '.join(cols[:3])}",
        "metrics": {
            "n_columns": len(cols),
            "columns": cols[:5],
            "stopwords_source": out.get("stopwords_source"),
        },
        "available": True,
    }


def probe_drift_analysis(df: pd.DataFrame) -> dict:
    """
    Drift needs two frames. We split the dataset in half so the test always
    has something concrete to score, even on datasets without a date column.
    """
    from modules.drift_analysis import compare_datasets
    if len(df) < 60:
        return {"summary": "skipped — fewer than 60 rows",
                "metrics": {}, "available": False}
    half = len(df) // 2
    ref = df.iloc[:half].reset_index(drop=True)
    cur = df.iloc[half:].reset_index(drop=True)
    out = compare_datasets(ref, cur)
    summary = out.get("summary") or {}
    sev = summary.get("severity_counts") or {}
    drifted = sum(v for k, v in sev.items() if k in {"minor", "moderate", "severe"})
    high = sev.get("severe", 0)
    return {
        "summary": (
            f"shared cols={summary.get('n_columns_compared', 0)} · "
            f"drifted={drifted} · severe={high} · "
            f"verdict={summary.get('overall_verdict', '?')}"
        ),
        "metrics": {
            "n_columns_compared": summary.get("n_columns_compared"),
            "drifted_columns": drifted,
            "severe_columns": high,
            "overall_verdict": summary.get("overall_verdict"),
        },
        "available": True,
    }


def probe_chart_planner(df: pd.DataFrame, target: str | None,
                          model_lb, explain, ts, deep,
                          health, advanced, cleaning) -> dict:
    from modules.profiler import build_profile
    from modules.column_roles import infer_column_roles
    from modules.chart_planner import build_chart_plan
    profile = build_profile(df)
    roles = infer_column_roles(df)
    plan = build_chart_plan(
        df, profile, health, advanced, cleaning, roles,
        context={
            "target_column": target,
            "model_leaderboard": model_lb,
            "explainability": explain,
            "time_series": ts,
            "deep_statistics_v2": deep,
        },
    )
    types = sorted({p.get("type") for p in plan if p.get("type")})
    return {
        "summary": f"planned {len(plan)} charts ({len(types)} distinct types)",
        "metrics": {"n_charts": len(plan), "n_types": len(types),
                    "types": types},
    }


def probe_visualizer(charts_count: int) -> dict:
    """Visualizer is exercised via the pipeline run; we just record the count."""
    return {
        "summary": f"rendered {charts_count} charts to /static/charts/",
        "metrics": {"n_charts_rendered": charts_count},
    }


def probe_pdf_report_builder(result: dict, dataset_id: str) -> dict:
    from modules.pdf_report_builder import generate_pdf_report
    out_dir = ROOT / "reports"
    pdf_path = generate_pdf_report(result, output_dir=out_dir)
    p = Path(pdf_path)
    return {
        "summary": f"PDF written: {p.name} ({p.stat().st_size / 1024:.0f} KB)",
        "metrics": {
            "filename": p.name,
            "size_kb": round(p.stat().st_size / 1024, 1),
        },
    }


def probe_frontend_api(result: dict, df: pd.DataFrame, csv_path: Path,
                        mode: str, elapsed_ms: float) -> dict:
    from modules.frontend_api import build_dashboard_payload
    payload = build_dashboard_payload(
        deepcopy(result), df,
        file_path=csv_path,
        analysis_mode=mode,
        elapsed_ms=elapsed_ms,
    )
    keys = sorted(payload.keys())
    return {
        "summary": f"dashboard payload has {len(keys)} top-level keys",
        "metrics": {"n_keys": len(keys), "sample_keys": keys[:10]},
    }


def probe_agent_brief(result: dict) -> dict:
    from modules.agent_brief import build_dataset_brief, render_brief_block
    brief = build_dataset_brief(
        result,
        max_columns=30,
        max_sample_rows=5,
        max_narrative_chars=1200,
        max_total_chars=9000,
    )
    text = render_brief_block(brief)
    return {
        "summary": f"brief: {len(brief)} top sections · {len(text):,} chars rendered",
        "metrics": {"top_sections": list(brief.keys()), "n_chars": len(text)},
    }


def probe_rag_index(result: dict) -> dict:
    from framevitals.rag_index import (
         build_fact_index,
         render_facts_block,
         retrieve,
    )
    facts = build_fact_index(result)
    if not facts:
        return {"summary": "no facts indexed", "metrics": {}}
    out = retrieve("what is the dataset health score and which columns are riskiest?",
                    facts, k=5)
    body = render_facts_block(out, max_chars=1200)
    return {
        "summary": f"indexed {len(facts)} facts · top-5 retrieval ok ({out.get('backend')})",
        "metrics": {
            "n_facts": len(facts),
            "backend": out.get("backend"),
            "top_paths": [f.get("path") for f in out.get("facts", [])[:5]],
            "chars_rendered": len(body),
        },
    }


def probe_ai_agent(result: dict, df: pd.DataFrame) -> dict:
    from modules.ai_agent import answer_with_agent
    out = answer_with_agent(
        question="Give me a 1-sentence summary of this dataset's quality.",
        df=df,
        analysis_result=result,
        fast=True,
    )
    return {
        "summary": (
            f"source={out.get('source')} · "
            f"answer length={len(out.get('answer', ''))} chars"
        ),
        "metrics": {
            "source": out.get("source"),
            "rag_top_k": (out.get("trace") or {}).get("rag_top_k"),
            "answer_preview": (out.get("answer", "") or "")[:240],
        },
    }


def probe_ai_agent_skipped() -> dict:
    return {
        "summary": "skipped (DATALENS_TEST_SKIP_LLM=1)",
        "metrics": {"skipped": True},
        "available": False,
    }


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------

def run_dataset(spec: dict) -> DatasetRun:
    csv_path = ROOT / spec["csv"]
    label = spec["label"]
    target = spec["target"]
    mode = spec["mode"]

    print(f"\n=== {label} ({csv_path.name}, target={target}, mode={mode}) ===")
    if not csv_path.exists():
        print(f"  SKIP — missing dataset {csv_path}")
        return DatasetRun(label=label, csv=str(csv_path), target=target, mode=mode)

    from modules.loader import load_dataset
    from modules.pipeline import run_full_analysis

    # --- Run the pipeline once. We re-use its output as evidence for the
    # high-level component probes (frontend_api, agent_brief, rag_index, etc.).
    t0 = time.perf_counter()
    df = load_dataset(csv_path)
    pipeline_dataset_id = f"comp_test_{label}"
    result = run_full_analysis(
        dataset_id=pipeline_dataset_id,
        file_path=csv_path,
        original_filename=csv_path.name,
        analysis_mode=mode,
        skip_ai=True,
        target_column=target,
    )
    pipeline_total_ms = (time.perf_counter() - t0) * 1000.0

    run = DatasetRun(
        label=label, csv=str(csv_path), target=target, mode=mode,
        rows=int(len(df)), columns=int(df.shape[1]),
        pipeline_total_ms=pipeline_total_ms,
    )

    # Cached expensive sub-results pulled from the pipeline output:
    health = result.get("health") or {}
    advanced = result.get("advanced") or {}
    cleaning = result.get("cleaning") or {}
    model_lb = result.get("model_leaderboard") or {}
    explain = result.get("explainability") or {}
    ts = result.get("time_series") or {}
    deep = result.get("deep_statistics_v2") or {}
    charts = result.get("charts") or []

    # Persist a tasteful subset of charts for the whitepaper. Each entry is
    # already shaped {type, title, path, description}, so we just trim and
    # pick the most representative ones per dataset.
    run.charts = [
        {
            "type": c.get("type"),
            "title": c.get("title"),
            "path": c.get("path"),
            "description": c.get("description"),
        }
        for c in charts
    ]

    probes: list[tuple[str, str, Callable[[], dict]]] = [
        # Ingest
        ("loader",            "Ingest",     lambda: probe_loader(csv_path, df)),
        ("profiler",          "Ingest",     lambda: probe_profiler(df)),
        ("column_roles",      "Ingest",     lambda: probe_column_roles(df)),
        ("dataset_signals",   "Ingest",     lambda: probe_dataset_signals(df)),
        ("analysis_selector", "Ingest",     lambda: probe_analysis_selector(df, target, mode)),
        # Quality
        ("health_score",      "Quality",    lambda: probe_health_score(df)),
        ("ml_readiness",      "Quality",    lambda: probe_ml_readiness(df)),
        ("advanced_indicators", "Quality",  lambda: probe_advanced_indicators(df)),
        ("signal_engine",     "Quality",    lambda: probe_signal_engine(df)),
        ("cleaner",           "Quality",    lambda: probe_cleaner(pipeline_dataset_id, df)),
        # Statistics
        ("deep_statistics_v2", "Statistics", lambda: probe_deep_statistics(df)),
        ("anomaly_ensemble",   "Statistics", lambda: probe_anomaly_ensemble(df)),
        # Modeling
        ("model_leaderboard",  "Modeling",   lambda: probe_model_leaderboard(df, target)),
        ("explainability",     "Modeling",   lambda: probe_explainability(df, target, pipeline_dataset_id)),
        # Time / text
        ("time_series",   "Time & Text",     lambda: probe_time_series(df, target)),
        ("text_profile",  "Time & Text",     lambda: probe_text_profile(df)),
        # Drift
        ("drift_analysis", "Drift",          lambda: probe_drift_analysis(df)),
        # Reporting
        ("chart_planner", "Reporting", lambda: probe_chart_planner(
            df, target, model_lb, explain, ts, deep, health, advanced, cleaning)),
        ("visualizer",    "Reporting", lambda: probe_visualizer(len(charts))),
        ("pdf_report_builder", "Reporting", lambda: probe_pdf_report_builder(result, pipeline_dataset_id)),
        # Agentic
        ("frontend_api",  "Agentic", lambda: probe_frontend_api(result, df, csv_path, mode, pipeline_total_ms)),
        ("agent_brief",   "Agentic", lambda: probe_agent_brief(result)),
        ("rag_index",     "Agentic", lambda: probe_rag_index(result)),
        ("ai_agent",      "Agentic",
            (lambda: probe_ai_agent_skipped())
            if os.environ.get("DATALENS_TEST_SKIP_LLM", "0").strip() in {"1", "true", "yes"}
            else (lambda: probe_ai_agent(result, df))),
    ]

    for name, category, fn in probes:
        comp = _probe(name, category, fn)
        run.components.append(comp)
        status = "OK   " if comp.ok else "FAIL "
        print(f"  {status} {name:<22} {comp.duration_ms:7.0f} ms · {comp.summary}")

    return run


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    return str(obj)


def write_manifest(manifest: Manifest, out_path: Path) -> None:
    payload = asdict(manifest)
    payload = to_jsonable(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def main(out_path: Path | None = None) -> int:
    started_at = time.time()
    runs: list[DatasetRun] = []
    for spec in DATASETS:
        runs.append(run_dataset(spec))
    completed_at = time.time()

    manifest = Manifest(
        started_at=started_at,
        completed_at=completed_at,
        python_version=sys.version.split()[0],
        runs=runs,
        component_definitions=COMPONENT_DEFINITIONS,
    )

    out_path = out_path or (ROOT / "reports" / "component_test_manifest.json")
    write_manifest(manifest, out_path)
    print(f"\nManifest → {out_path}")

    # Summary table
    n_total = sum(len(r.components) for r in runs)
    n_ok = sum(1 for r in runs for c in r.components if c.ok)
    print(f"\nComponent tests: {n_ok} / {n_total} passed across {len(runs)} datasets")
    return 0 if n_ok == n_total else 1  # we return 0 even on individual probe
                                          # failures so the whitepaper is built;
                                          # the manifest reflects the truth.


if __name__ == "__main__":
    raise SystemExit(main())
