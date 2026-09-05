"""FrameVitals analysis pipeline.

Runs the analytics stack in phases and parallelizes independent analyses where
safe. Optional failures are converted into structured error payloads so a
single diagnostic does not sink the whole report. Expensive modules can be
explicitly disabled through ``AnalysisConfig`` while adaptive execution budgets
bound dangerous work on large or wide datasets.
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
from framevitals.budgeted_analysis import (
    run_budgeted_anomalies,
    run_budgeted_deep_statistics,
    run_budgeted_time_series,
)
from framevitals.cleaner import create_cleaned_dataset
from framevitals.column_roles import infer_column_roles, summarize_roles
from framevitals.config import VALID_MODULES
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.execution import derive_execution_budget
from framevitals.health_score import calculate_health_score
from framevitals.loader import load_dataset
from framevitals.ml_readiness import calculate_ml_readiness
from framevitals.model_leaderboard import run_model_leaderboard
from framevitals.profiler import build_profile
from framevitals.quality_diagnostics import run_quality_diagnostics
from framevitals.signal_engine import build_signals
from framevitals.target_intelligence import run_target_intelligence
from framevitals.text_profile import profile_text_columns


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


def _skipped_module(module: str, reason: str = "Disabled by configuration.") -> dict:
    return {
        "available": False,
        "skipped": True,
        "module": module,
        "reason": reason,
    }


def _result_status(value: Any) -> str:
    if isinstance(value, dict) and value.get("error"):
        return "error"
    return "ran"


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
    disabled_modules: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict:
    """Run the complete FrameVitals analysis pipeline."""
    overall_start = time.perf_counter()
    timings_ms: dict[str, Any] = {}

    disabled = set(disabled_modules or ())
    unknown_modules = sorted(disabled - VALID_MODULES)
    if unknown_modules:
        raise ValueError(
            "Unknown disabled module(s): " + ", ".join(unknown_modules)
        )
    module_status: dict[str, str] = {
        name: "pending" for name in sorted(VALID_MODULES)
    }

    def module_enabled(name: str) -> bool:
        return name not in disabled

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

    execution_budget = derive_execution_budget(
        len(df),
        len(df.columns),
        mode=analysis_mode,
    )

    t0 = time.perf_counter()
    profile = build_profile(df)
    timings_ms["profile"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    column_roles = infer_column_roles(df)
    roles_summary = summarize_roles(column_roles)
    timings_ms["column_roles"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    dataset_signals = detect_dataset_signals(
        df,
        profile,
        column_roles=column_roles,
    )
    timings_ms["dataset_signals"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    analysis_selection = select_analyses(
        signals=dataset_signals,
        analysis_mode=analysis_mode,
        target_column=target_column,
    )
    analysis_selection["execution_modules"] = {
        "disabled": sorted(disabled),
        "enabled": sorted(VALID_MODULES - disabled),
    }
    analysis_selection["execution_budget"] = execution_budget.to_dict()
    timings_ms["analysis_selection"] = (time.perf_counter() - t0) * 1000

    # Phase 2: core quality/readiness plus bounded practical diagnostics.
    t0 = time.perf_counter()
    health = calculate_health_score(df, profile)
    timings_ms["health"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ml_readiness = calculate_ml_readiness(df, profile=profile)
    timings_ms["ml_readiness"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    advanced = calculate_advanced_indicators(df)
    timings_ms["advanced"] = (time.perf_counter() - t0) * 1000

    if module_enabled("quality_diagnostics"):
        _, quality_diagnostics, quality_elapsed = _safe_call(
            "quality_diagnostics",
            lambda: run_quality_diagnostics(
                df,
                profile=profile,
                column_roles=column_roles,
                max_sample_rows=execution_budget.quality_sample_rows,
            ),
        )
        timings_ms["quality_diagnostics"] = quality_elapsed
        module_status["quality_diagnostics"] = _result_status(quality_diagnostics)
    else:
        quality_diagnostics = _skipped_module("quality_diagnostics")
        timings_ms["quality_diagnostics"] = 0.0
        module_status["quality_diagnostics"] = "disabled"

    # Phase 3: independent heavier analyses. Legacy algorithms are routed
    # through bounded adapters until their streaming/native replacements land.
    deep_statistics_v2 = None
    anomalies_v2 = None
    time_series_analysis = None
    text_profile = None

    phase3_modules: list[tuple[str, str, Callable[[], Any]]] = [
        (
            "deep_statistics",
            "deep_statistics_v2",
            lambda: run_budgeted_deep_statistics(df, budget=execution_budget),
        ),
        (
            "anomaly_detection",
            "anomalies_v2",
            lambda: run_budgeted_anomalies(df, budget=execution_budget),
        ),
        (
            "time_series",
            "time_series",
            lambda: run_budgeted_time_series(
                df,
                budget=execution_budget,
                target_column=target_column,
            ),
        ),
        ("text_profile", "text_profile", lambda: profile_text_columns(df)),
    ]

    phase3_results: dict[str, Any] = {}
    phase3_worker_limit = min(
        parallel_workers,
        execution_budget.max_memory_heavy_parallelism,
    )
    if analysis_mode in {"standard", "deep", "research"}:
        tasks: list[tuple[str, str, Callable[[], Any]]] = []
        for module, result_key, fn in phase3_modules:
            if module_enabled(module):
                tasks.append((module, result_key, fn))
                module_status[module] = "scheduled"
            else:
                phase3_results[result_key] = _skipped_module(module)
                module_status[module] = "disabled"

        per_task_ms: dict[str, float] = {}
        if tasks:
            phase3_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=phase3_worker_limit) as executor:
                futures = {
                    executor.submit(_safe_call, result_key, fn): (module, result_key)
                    for module, result_key, fn in tasks
                }
                for future in as_completed(futures):
                    module, result_key = futures[future]
                    name, value, elapsed = future.result()
                    phase3_results[result_key] = value
                    per_task_ms[name] = elapsed
                    module_status[module] = _result_status(value)

            timings_ms["phase3_parallel_total"] = (
                time.perf_counter() - phase3_start
            ) * 1000
        else:
            timings_ms["phase3_parallel_total"] = 0.0
        timings_ms["phase3_tasks"] = per_task_ms
    else:
        for module, _, _ in phase3_modules:
            module_status[module] = "not_applicable"

    deep_statistics_v2 = phase3_results.get("deep_statistics_v2")
    anomalies_v2 = phase3_results.get("anomalies_v2")
    time_series_analysis = phase3_results.get("time_series")
    text_profile = phase3_results.get("text_profile")

    # Phase 4: target-aware diagnostics and ML chain.
    target_intelligence = None
    model_leaderboard = None
    explainability = None

    if target_column:
        if module_enabled("target_intelligence"):
            t0 = time.perf_counter()
            _, target_intelligence, _ = _safe_call(
                "target_intelligence",
                lambda: run_target_intelligence(
                    df,
                    target_column=target_column,
                    column_roles=column_roles,
                ),
            )
            timings_ms["target_intelligence"] = (time.perf_counter() - t0) * 1000
            module_status["target_intelligence"] = _result_status(target_intelligence)
        else:
            target_intelligence = _skipped_module("target_intelligence")
            timings_ms["target_intelligence"] = 0.0
            module_status["target_intelligence"] = "disabled"
    else:
        module_status["target_intelligence"] = "not_applicable"

    modeling_applicable = target_column and analysis_mode in {"standard", "deep", "research"}
    if modeling_applicable:
        if module_enabled("modeling"):
            t0 = time.perf_counter()
            _, model_leaderboard, _ = _safe_call(
                "model_leaderboard",
                lambda: run_model_leaderboard(df, target_column=target_column),
            )
            timings_ms["model_leaderboard"] = (time.perf_counter() - t0) * 1000
            module_status["modeling"] = _result_status(model_leaderboard)
        else:
            model_leaderboard = _skipped_module("modeling")
            timings_ms["model_leaderboard"] = 0.0
            module_status["modeling"] = "disabled"
    else:
        module_status["modeling"] = "not_applicable"

    winner_available = (
        isinstance(model_leaderboard, dict)
        and model_leaderboard.get("available")
        and model_leaderboard.get("winner")
    )
    if winner_available:
        if module_enabled("explainability"):
            t0 = time.perf_counter()

            def run_explainability():
                from framevitals.explainability import explain_winner

                return explain_winner(
                    df,
                    target_column=target_column,
                    leaderboard_result=model_leaderboard,
                    dataset_id=dataset_id,
                )

            _, explainability, _ = _safe_call(
                "explainability",
                run_explainability,
            )
            timings_ms["explainability"] = (time.perf_counter() - t0) * 1000
            module_status["explainability"] = _result_status(explainability)
        else:
            explainability = _skipped_module("explainability")
            timings_ms["explainability"] = 0.0
            module_status["explainability"] = "disabled"
    elif not module_enabled("explainability"):
        explainability = _skipped_module("explainability")
        module_status["explainability"] = "disabled"
    else:
        module_status["explainability"] = "not_applicable"

    # Phase 5: signals, cleaning, and optional visual artifacts.
    t0 = time.perf_counter()
    signals = build_signals(profile, health, ml_readiness, advanced)
    timings_ms["signals"] = (time.perf_counter() - t0) * 1000

    if module_enabled("cleaning"):
        t0 = time.perf_counter()
        cleaning = create_cleaned_dataset(
            dataset_id,
            df,
            write_output=write_artifacts,
            before_profile=profile,
            before_health=health,
        )
        timings_ms["cleaning"] = (time.perf_counter() - t0) * 1000
        module_status["cleaning"] = "ran"
    else:
        missing_count = sum(
            int(value)
            for value in profile.get("missing_counts", {}).values()
            if value is not None
        )
        duplicate_count = int(profile.get("duplicate_rows", 0) or 0)
        cleaning = {
            **_skipped_module("cleaning"),
            "actions": [],
            "before_health": health,
            "after_health": health,
            "output_path": None,
            "missing_before": missing_count,
            "missing_after": missing_count,
            "duplicates_before": duplicate_count,
            "duplicates_after": duplicate_count,
        }
        timings_ms["cleaning"] = 0.0
        module_status["cleaning"] = "disabled"

    charts: list[dict] = []
    charts_applicable = write_artifacts and analysis_mode in {"standard", "deep", "research"}
    if charts_applicable and module_enabled("charts"):
        t0 = time.perf_counter()

        def render_charts():
            from framevitals.visualizer import generate_charts

            return generate_charts(
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

        _, rendered_charts, chart_elapsed = _safe_call("charts", render_charts)
        timings_ms["charts"] = chart_elapsed
        if isinstance(rendered_charts, list):
            charts = rendered_charts
            module_status["charts"] = "ran"
        else:
            charts = []
            module_status["charts"] = _result_status(rendered_charts)
    elif not module_enabled("charts"):
        timings_ms["charts"] = 0.0
        module_status["charts"] = "disabled"
    else:
        timings_ms["charts"] = 0.0
        module_status["charts"] = "not_applicable"

    # Phase 6: optional AI interpretation.
    ai_env = os.environ.get(
        "FRAMEVITALS_ANALYZE_AI",
        os.environ.get("DATALENS_ANALYZE_AI", "0"),
    )
    analyze_ai_default = ai_env.strip().lower() in {"1", "true", "yes"}

    if not module_enabled("ai"):
        ai_report = {
            "source": "disabled",
            "text": "AI report disabled by configuration.",
            "deferred": False,
        }
        timings_ms["ai_report"] = 0.0
        module_status["ai"] = "disabled"
    elif skip_ai or not analyze_ai_default:
        ai_report = {
            "source": "deferred" if not skip_ai else "skipped",
            "text": "" if not skip_ai else "AI report skipped.",
            "deferred": not skip_ai,
        }
        timings_ms["ai_report"] = 0.0
        module_status["ai"] = "deferred" if not skip_ai else "skipped_by_caller"
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
            module_status["ai"] = "ran"
        except Exception as exc:  # noqa: BLE001
            logger.exception("AI report generation failed")
            ai_report = {"source": f"error: {exc}", "text": str(exc)}
            module_status["ai"] = "error"
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
        "execution": {
            "disabled_modules": sorted(disabled),
            "module_status": module_status,
            "budget": execution_budget.to_dict(),
            "phase3_worker_limit": int(phase3_worker_limit),
        },
        "profile": profile,
        "column_roles": column_roles,
        "roles_summary": roles_summary,
        "dataset_signals": dataset_signals,
        "analysis_selection": analysis_selection,
        "health": health,
        "signals": signals,
        "ml_readiness": ml_readiness,
        "advanced": advanced,
        "quality_diagnostics": quality_diagnostics,
        "deep_statistics_v2": deep_statistics_v2,
        "anomalies_v2": anomalies_v2,
        "target_intelligence": target_intelligence,
        "model_leaderboard": model_leaderboard,
        "explainability": explainability,
        "time_series": time_series_analysis,
        "text_profile": text_profile,
        "cleaning": cleaning,
        "charts": charts,
        "ai_report": ai_report,
        "timings_ms": timings_ms,
    }
