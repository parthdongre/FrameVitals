"""Planning-only public execution path.

`fv.plan()` should be cheap enough to call before committing to a full analysis.
This module deliberately avoids importing the full FrameVitals pipeline while
still resolving configuration, structural signals, and adaptive execution
budgets exactly as the execution layer expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from framevitals.analysis_selector import select_analyses
from framevitals.column_roles import infer_column_roles
from framevitals.config import ConfigInput, VALID_MODULES, resolve_config
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.execution import derive_execution_budget
from framevitals.planning import AnalysisPlan
from framevitals.profiler import build_profile
from framevitals.sources import resolve_source


DataInput = str | Path | pd.DataFrame


def plan(
    data: DataInput,
    *,
    target: str | None = None,
    mode: str | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: ConfigInput = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
) -> AnalysisPlan:
    """Preview planned analyses, scale policy, and execution constraints."""
    resolved = resolve_config(
        config,
        preset=preset,
        mode=mode,
        target=target,
        workers=workers,
        artifacts=False,
        disabled_modules=disabled_modules,
    )

    source = resolve_source(data)
    source_metadata = source.inspect()
    dataframe = source.load()

    if resolved.target is not None and resolved.target not in dataframe.columns:
        raise ValueError(f"Target column not found: {resolved.target}")

    dataset_profile = build_profile(dataframe)
    column_roles = infer_column_roles(dataframe)
    dataset_signals = detect_dataset_signals(
        dataframe,
        dataset_profile,
        column_roles=column_roles,
    )

    budget = derive_execution_budget(
        len(dataframe),
        len(dataframe.columns),
        mode=resolved.mode,
    )

    selection = select_analyses(
        signals=dataset_signals,
        analysis_mode=resolved.mode,
        target_column=resolved.target,
    )
    disabled = set(resolved.disabled_modules)
    selection["execution_modules"] = {
        "disabled": sorted(disabled),
        "enabled": sorted(VALID_MODULES - disabled),
    }
    selection["execution_budget"] = budget.to_dict()

    public_signals = {
        key: value
        for key, value in dataset_signals.items()
        if key != "column_roles"
    }

    return AnalysisPlan({
        "dataset_name": source_metadata.name,
        "source": source_metadata.to_dict(),
        "analysis_mode": resolved.mode,
        "target": resolved.target,
        "shape": dict(dataset_profile.get("shape", {})),
        "config": resolved.to_dict(),
        "execution_budget": budget.to_dict(),
        "signals": public_signals,
        "selection": selection,
    })
