"""Bounded module scheduler for streaming analyses.

Streaming sources already own profile, role, health, readiness, quality, and
source-shape state. Re-entering the materialized pipeline on the retained sample
would recompute those facts and then throw them away. This scheduler runs only
modules that genuinely require row-level values while consuming the source-level
execution budget chosen by the streaming planner.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from framevitals.advanced_indicators import calculate_advanced_indicators
from framevitals.budgeted_analysis import (
    run_budgeted_anomalies,
    run_budgeted_deep_statistics,
    run_budgeted_time_series,
)
from framevitals.config import VALID_MODULES
from framevitals.execution import ExecutionBudget, derive_execution_budget
from framevitals.model_leaderboard import run_model_leaderboard
from framevitals.pipeline import _result_status, _safe_call, _skipped_module
from framevitals.target_intelligence import run_target_intelligence
from framevitals.text_profile import profile_text_columns


def run_streaming_bounded_modules(
    dataframe: pd.DataFrame,
    *,
    dataset_id: str,
    original_filename: str,
    analysis_mode: str,
    target_column: str | None,
    parallel_workers: int,
    source_budget: ExecutionBudget,
    column_roles: dict[str, Any],
    skip_ai: bool,
    disabled_modules: set[str] | tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Run only row-dependent modules on the bounded streaming sample.

    Statistical adapters receive ``source_budget`` so ultra-wide source limits
    remain authoritative even though the retained sample is much narrower. The
    scheduler's concurrency limit, however, is derived from the bounded sample
    because that is the memory footprint actually resident during parallel work.
    """
    overall_start = time.perf_counter()
    timings_ms: dict[str, Any] = {}
    disabled = set(disabled_modules or ())
    unknown = sorted(disabled - VALID_MODULES)
    if unknown:
        raise ValueError("Unknown disabled module(s): " + ", ".join(unknown))
    if dataframe.empty:
        raise ValueError("Bounded streaming sample is empty.")
    if target_column is not None and target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")

    module_status: dict[str, str] = {
        name: "pending" for name in sorted(VALID_MODULES)
    }

    def module_enabled(name: str) -> bool:
        return name not in disabled

    t0 = time.perf_counter()
    advanced = calculate_advanced_indicators(dataframe)
    timings_ms["advanced"] = (time.perf_counter() - t0) * 1000

    phase3_modules: list[tuple[str, str, Callable[[], Any]]] = [
        (
            "deep_statistics",
            "deep_statistics_v2",
            lambda: run_budgeted_deep_statistics(dataframe, budget=source_budget),
        ),
        (
            "anomaly_detection",
            "anomalies_v2",
            lambda: run_budgeted_anomalies(dataframe, budget=source_budget),
        ),
        (
            "time_series",
            "time_series",
            lambda: run_budgeted_time_series(
                dataframe,
                budget=source_budget,
                target_column=target_column,
            ),
        ),
        ("text_profile", "text_profile", lambda: profile_text_columns(dataframe)),
    ]

    phase3_results: dict[str, Any] = {}
    bounded_parallel_budget = derive_execution_budget(
        len(dataframe),
        len(dataframe.columns),
        mode=analysis_mode,
    )
    phase3_worker_limit = max(
        1,
        min(
            int(parallel_workers),
            int(bounded_parallel_budget.max_memory_heavy_parallelism),
        ),
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
        timings_ms["phase3_parallel_total"] = 0.0
        timings_ms["phase3_tasks"] = {}
        for module, _, _ in phase3_modules:
            module_status[module] = "not_applicable"

    deep_statistics_v2 = phase3_results.get("deep_statistics_v2")
    anomalies_v2 = phase3_results.get("anomalies_v2")
    time_series_analysis = phase3_results.get("time_series")
    text_profile = phase3_results.get("text_profile")

    target_intelligence = None
    model_leaderboard = None
    explainability = None

    if target_column:
        if module_enabled("target_intelligence"):
            t0 = time.perf_counter()
            _, target_intelligence, _ = _safe_call(
                "target_intelligence",
                lambda: run_target_intelligence(
                    dataframe,
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

    modeling_applicable = bool(
        target_column and analysis_mode in {"standard", "deep", "research"}
    )
    if modeling_applicable:
        if module_enabled("modeling"):
            t0 = time.perf_counter()
            _, model_leaderboard, _ = _safe_call(
                "model_leaderboard",
                lambda: run_model_leaderboard(dataframe, target_column=target_column),
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
                    dataframe,
                    target_column=target_column,
                    leaderboard_result=model_leaderboard,
                    dataset_id=dataset_id,
                )

            _, explainability, _ = _safe_call("explainability", run_explainability)
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

    # Full AI generation consumes the final full-stream profile/signals and is
    # therefore intentionally left to the outer streaming orchestrator. The
    # public analyze API currently requests skip_ai=True on this path.
    if not module_enabled("ai"):
        ai_report = {
            "source": "disabled",
            "text": "AI report disabled by configuration.",
            "deferred": False,
        }
        module_status["ai"] = "disabled"
    elif skip_ai:
        ai_report = {
            "source": "skipped",
            "text": "AI report skipped.",
            "deferred": False,
        }
        module_status["ai"] = "skipped_by_caller"
    else:
        ai_env = os.environ.get(
            "FRAMEVITALS_ANALYZE_AI",
            os.environ.get("DATALENS_ANALYZE_AI", "0"),
        )
        enabled = ai_env.strip().lower() in {"1", "true", "yes"}
        ai_report = {
            "source": "deferred",
            "text": "",
            "deferred": True,
            "reason": (
                "Streaming AI interpretation is deferred until full-stream core "
                "signals are assembled."
                if enabled
                else "AI analysis is not enabled by environment."
            ),
        }
        module_status["ai"] = "deferred"
    timings_ms["ai_report"] = 0.0

    # Core streaming modules are intentionally not executed here. Their status
    # is filled by the outer streaming pipeline after reuse.
    for module in ("quality_diagnostics", "cleaning", "charts"):
        if module in disabled:
            module_status[module] = "disabled"

    timings_ms["total"] = (time.perf_counter() - overall_start) * 1000
    return {
        "dataset_id": dataset_id,
        "filename": original_filename,
        "analysis_mode": analysis_mode,
        "artifacts_enabled": False,
        "execution": {
            "disabled_modules": sorted(disabled),
            "module_status": module_status,
            "budget": source_budget.to_dict(),
            "phase3_worker_limit": int(phase3_worker_limit),
            "bounded_scheduler": {
                "enabled": True,
                "core_reprofiled": False,
                "source_budget_scope": "source_shape",
                "parallelism_budget_scope": "bounded_sample",
                "sample_rows": int(len(dataframe)),
                "sample_columns": int(len(dataframe.columns)),
            },
        },
        "advanced": advanced,
        "deep_statistics_v2": deep_statistics_v2,
        "anomalies_v2": anomalies_v2,
        "target_intelligence": target_intelligence,
        "model_leaderboard": model_leaderboard,
        "explainability": explainability,
        "time_series": time_series_analysis,
        "text_profile": text_profile,
        "ai_report": ai_report,
        "timings_ms": timings_ms,
    }
