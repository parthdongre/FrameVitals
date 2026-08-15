"""FrameVitals analysis pipeline.

Runs the full analytics stack in phases and parallelizes independent analyses
where safe. Optional failures are converted into structured error payloads so a
single diagnostic does not sink the whole report.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from framevitals.advanced_indicators import calculate_advanced_indicators
from framevitals.ai_insights import generate_ai_report
from framevitals.analysis_selector import select_analyses
from framevitals.anomaly_ensemble import detect_anomalies_ensemble
from framevitals.cleaner import create_cleaned_dataset
from framevitals.column_roles import infer_column_roles, summarize_roles
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.deep_statistics_v2 import run_deep_statistics_v2
from framevitals.explainability import explain_winner
from framevitals.health_score import calculate_health_score
from framevitals.loader import load_dataset
from framevitals.ml_readiness import calculate_ml_readiness
from framevitals.model_leaderboard import run_model_leaderboard
from framevitals.profiler import build_profile
from framevitals.signal_engine import build_signals
from framevitals.text_profile import profile_text_columns
from framevitals.time_series import detect_and_analyze_time_series
from framevitals.visualizer import generate_charts


logger = logging.getLogger("framevitals.pipeline")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _safe_call(name: str, fn: Callable[[], Any]) -> tuple[str, Any, float]:
    """Run a callable, time it, and convert failures into JSON-safe results."""
    t0 = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - optional analyses should fail soft
        logger.exception("pipeline task %s failed", name)
        value = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return name, value, elapsed_ms


def run_full_analysis(
    dataset_id: str,
    file_path=None,
    original_filename: str = "<dataframe>",
    analysis_mode: str = "standard",
    skip_ai: bool = False,
    target_column: str | None = None,
    parallel_workers: int = 4,
    dataframe: pd.DataFrame | None = None,
    write_artifacts: bool = True,
) -> dict:
    """Run the complete FrameVitals analysis pipeline.

    Parameters
    ----------
    dataset_id:
        Unique identifier used for optional artifact filenames.
    file_path:
        Path-like dataset source. Omit when ``dataframe`` is supplied.
    original_filename:
        Source label included in the result.
    analysis_mode:
        One of ``quick``, ``standard``, ``deep``, or ``research``.
    skip_ai:
        Skip the optional LLM report.
    target_column:
        Optional supervised-learning target column.
    parallel_workers:
        Worker count for independent diagnostic tasks.
    dataframe:
        Optional in-memory DataFrame. When supplied it takes precedence over
        ``file_path`` and is copied before analysis.
    write_artifacts:
        Persist cleaned CSV/chart artifacts. Web application callers keep this
        enabled; reusable library calls can disable filesystem side effects.
    """
    overall_start = time.perf_counter()
    timings_ms: dict[str, Any] = {}

    # Phase 1: load and understand structure.
    t0 = time.perf_counter()
    if dataframe is not None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        df = dataframe.copy()
    elif file_path is not None:
        df = load_dataset(file_path)
    else:
        raise ValueError("Provide either file_path or dataframe.")
    timings_ms["load"] = (time.perf_counter() - t0) * 1000

    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    t0 = time.perf_counter()
    profile = build_profile(df)
    timings_ms["profile"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    column_roles = infer_column_roles(df)
    roles_summary = summarize_roles(column_roles)
    timings_ms["column_roles"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dataset_signals = detect_dataset_signals(df, profile)
    timings_ms["dataset_signals"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    analysis_selection = select_analyses(
        signals=dataset_signals,
        analysis_mode=analysis_mode,
        target_column=target_column,
    )
    timings_ms["analysis_selection"] = (time.perf_counter() - t0) * 1000

    # Phase 2: quality scoring.
    t0 = time.perf_counter()
    health = calculate_health_score(df, profile)
    timings_ms["health"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ml_readiness = calculate_ml_readiness(df, profile=profile)
    timings_ms["ml_readiness"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    advanced = calculate_advanced_indicators(df)
    timings_ms["advanced"] = (time.perf_counter() - t0) * 1000

    # Phase 3: independent heavier analyses.
    deep_statistics_v2 = None
    anomalies_v2 = None
    time_series_analysis = None
    text_profile = None

    if analysis_mode in {"standard", "deep", "research"}:
        tasks: list[tuple[str, Callable[[], Any]]] = [
            ("deep_statistics_v2", lambda: run_deep_statistics_v2(df)),
            ("anomalies_v2", lambda: detect_anomalies_ensemble(df)),
            (
                "time_series",
                lambda: detect_and_analyze_time_series(
                    df,
                    target_column=target_column,
                ),
            ),
            ("text_profile", lambda: profile_text_columns(df)),
        ]

        results: dict[str, Any] = {}
        per_task_ms: dict[str, float] = {}

        phase3_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_safe_call, name, fn): name for name, fn in tasks
            }
            for future in as_completed(futures):
                name, value, elapsed = future.result()
                results[name] = value
                per_task_ms[name] = elapsed

        deep_statistics_v2 = results.get("deep_statistics_v2")
        anomalies_v2 = results.get("anomalies_v2")
        time_series_analysis = results.get("time_series")
        text_profile = results.get("text_profile")

        timings_ms["phase3_parallel_total"] = (
            time.perf_counter() - phase3_start
        ) * 1000
        timings_ms["phase3_tasks"] = per_task_ms

    # Phase 4: target-aware ML chain.
    model_leaderboard = None
    explainability = None
    if target_column and analysis_mode in {"standard", "deep", "research"}:
        t0 = time.perf_counter()
        _, model_leaderboard, _ = _safe_call(
            "model_leaderboard",
            lambda: run_model_leaderboard(df, target_column=target_column),
        )
        timings_ms["model_leaderboard"] = (time.perf_counter() - t0) * 1000

        if (
            isinstance(model_leaderboard, dict)
            and model_leaderboard.get("available")
            and model_leaderboard.get("winner")
        ):
            t0 = time.perf_counter()
            _, explainability, _ = _safe_call(
                "explainability",
                lambda: explain_winner(
                    df,
                    target_column=target_column,
                    leaderboard_result=model_leaderboard,
                    dataset_id=dataset_id,
                ),
            )
            timings_ms["explainability"] = (time.perf_counter() - t0) * 1000

    # Phase 5: signals, cleaning, and optional visual artifacts.
    t0 = time.perf_counter()
    signals = build_signals(profile, health, ml_readiness, advanced)
    timings_ms["signals"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cleaning = create_cleaned_dataset(
        dataset_id,
        df,
        write_output=write_artifacts,
        before_profile=profile,
        before_health=health,
    )
    timings_ms["cleaning"] = (time.perf_counter() - t0) * 1000

    charts: list[dict] = []
    if write_artifacts and analysis_mode in {"standard", "deep", "research"}:
        t0 = time.perf_counter()
        charts = generate_charts(
            dataset_id,
            df,
            health,
            advanced,
            cleaning,
            target_column=target_column,
            model_leaderboard=model_leaderboard,
            explainability=explainability,
            time_series=time_series_analysis,
            deep_statistics_v2=deep_statistics_v2,
        )
        timings_ms["charts"] = (time.perf_counter() - t0) * 1000

    # Phase 6: optional AI interpretation.
    ai_env = os.environ.get(
        "FRAMEVITALS_ANALYZE_AI",
        os.environ.get("DATALENS_ANALYZE_AI", "0"),
    )
    analyze_ai_default = ai_env.strip().lower() in {"1", "true", "yes"}

    if skip_ai or not analyze_ai_default:
        ai_report = {
            "source": "deferred" if not skip_ai else "skipped",
            "text": "" if not skip_ai else "AI report skipped.",
            "deferred": not skip_ai,
        }
        timings_ms["ai_report"] = 0.0
    else:
        t0 = time.perf_counter()
        try:
            ai_report = generate_ai_report(
                profile=profile,
                health=health,
                signals=signals,
                ml_readiness=ml_readiness,
                advanced=advanced,
                column_roles_summary=roles_summary,
                dataset_signals=dataset_signals,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("AI report generation failed")
            ai_report = {"source": f"error: {exc}", "text": str(exc)}
        timings_ms["ai_report"] = (time.perf_counter() - t0) * 1000

    timings_ms["total"] = (time.perf_counter() - overall_start) * 1000

    logger.info(
        "pipeline complete dataset=%s mode=%s target=%s total_ms=%.0f",
        dataset_id,
        analysis_mode,
        target_column,
        timings_ms["total"],
    )

    return {
        "dataset_id": dataset_id,
        "filename": original_filename,
        "analysis_mode": analysis_mode,
        "artifacts_enabled": write_artifacts,
        "profile": profile,
        "column_roles": column_roles,
        "roles_summary": roles_summary,
        "dataset_signals": dataset_signals,
        "analysis_selection": analysis_selection,
        "health": health,
        "signals": signals,
        "ml_readiness": ml_readiness,
        "advanced": advanced,
        "deep_statistics_v2": deep_statistics_v2,
        "anomalies_v2": anomalies_v2,
        "model_leaderboard": model_leaderboard,
        "explainability": explainability,
        "time_series": time_series_analysis,
        "text_profile": text_profile,
        "cleaning": cleaning,
        "charts": charts,
        "ai_report": ai_report,
        "timings_ms": timings_ms,
    }
