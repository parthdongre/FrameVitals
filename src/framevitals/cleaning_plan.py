"""Explicit, conservative cleaning plans for FrameVitals datasets.

Planning never mutates user data. Applying a plan returns a copy by default and
supports only the conservative operations already used by FrameVitals' internal
cleaner: duplicate removal and missing-value imputation by median/mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from framevitals.health_score import calculate_health_score
from framevitals.profiler import build_profile


CLEANING_PLAN_SCHEMA_VERSION = "1"


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


class CleaningPlan(dict):
    """Dict-compatible, inspectable cleaning plan."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def actions(self) -> list[dict[str, Any]]:
        value = self.get("actions", [])
        return value if isinstance(value, list) else []

    def summary(self) -> dict[str, Any]:
        risk_counts: dict[str, int] = {}
        for action in self.actions:
            risk = str(action.get("risk") or "unknown").lower()
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        return {
            "action_count": len(self.actions),
            "risk_counts": risk_counts,
            "duplicates_to_remove": self.get("duplicates_to_remove", 0),
            "missing_values_to_fill": self.get("missing_values_to_fill", 0),
        }

    def to_json(
        self,
        destination: str | Path | None = None,
        *,
        indent: int = 2,
    ) -> str | Path:
        rendered = json.dumps(dict(self), indent=indent, default=str)
        if destination is None:
            return rendered
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        return path

    def apply(self, dataframe: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
        return apply_cleaning_plan(dataframe, self, copy=copy)

    def simulate(
        self,
        dataframe: pd.DataFrame,
        *,
        before_profile: dict | None = None,
        before_health: dict | None = None,
    ) -> dict[str, Any]:
        return simulate_cleaning_plan(
            dataframe,
            self,
            before_profile=before_profile,
            before_health=before_health,
        )


def infer_cleaning_plan(
    dataframe: pd.DataFrame,
    *,
    profile: dict | None = None,
) -> CleaningPlan:
    """Infer the same conservative operations used by the internal cleaner."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if dataframe.empty:
        raise ValueError("Cannot infer a cleaning plan for an empty DataFrame.")

    if profile is None:
        profile = build_profile(dataframe)

    duplicate_count = int(profile.get("duplicate_rows", dataframe.duplicated().sum()))
    working = dataframe.drop_duplicates() if duplicate_count else dataframe
    actions: list[dict[str, Any]] = []

    if duplicate_count:
        actions.append({
            "id": "remove_duplicates",
            "type": "remove_duplicates",
            "column": None,
            "strategy": "drop_duplicate_rows",
            "affected": duplicate_count,
            "risk": "low",
            "description": f"Remove {duplicate_count} duplicate rows.",
        })

    missing_values_to_fill = 0
    missing_counts = working.isna().sum()
    for column, missing_value in missing_counts.items():
        missing = int(missing_value)
        if missing == 0:
            continue

        missing_values_to_fill += missing
        if pd.api.types.is_numeric_dtype(working[column]):
            fill_value = working[column].median()
            strategy = "median"
            action_type = "fill_numeric_missing"
            description = (
                f"Fill {missing} missing values in '{column}' using the median."
            )
        else:
            mode = working[column].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else "Unknown"
            strategy = "mode" if len(mode) else "constant"
            action_type = "fill_categorical_missing"
            description = (
                f"Fill {missing} missing values in '{column}' using "
                f"{'the mode' if len(mode) else 'a fallback value'}."
            )

        actions.append({
            "id": f"{action_type}:{column}",
            "type": action_type,
            "column": str(column),
            "strategy": strategy,
            "fill_value": _python_scalar(fill_value),
            "affected": missing,
            "risk": "medium",
            "description": description,
        })

    return CleaningPlan({
        "schema_version": CLEANING_PLAN_SCHEMA_VERSION,
        "actions": actions,
        "duplicates_to_remove": duplicate_count,
        "missing_values_to_fill": missing_values_to_fill,
    })


def apply_cleaning_plan(
    dataframe: pd.DataFrame,
    plan: Mapping[str, Any],
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Apply a validated FrameVitals cleaning plan and return the result."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a cleaning-plan mapping")
    if plan.get("schema_version") != CLEANING_PLAN_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported cleaning plan schema version: "
            f"{plan.get('schema_version')!r}"
        )

    actions = plan.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("Cleaning plan actions must be a list.")

    cleaned = dataframe.copy() if copy else dataframe
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError("Each cleaning plan action must be an object.")

        action_type = action.get("type")
        if action_type == "remove_duplicates":
            cleaned.drop_duplicates(inplace=True)
            continue

        if action_type in {"fill_numeric_missing", "fill_categorical_missing"}:
            column = action.get("column")
            if not isinstance(column, str) or column not in cleaned.columns:
                raise ValueError(f"Cleaning plan column not found: {column!r}")
            cleaned[column] = cleaned[column].fillna(action.get("fill_value"))
            continue

        raise ValueError(f"Unsupported cleaning action type: {action_type!r}")

    return cleaned


def simulate_cleaning_plan(
    dataframe: pd.DataFrame,
    plan: Mapping[str, Any],
    *,
    before_profile: dict | None = None,
    before_health: dict | None = None,
) -> dict[str, Any]:
    """Apply a plan to a copy and report expected quality/shape changes."""
    if before_profile is None:
        before_profile = build_profile(dataframe)
    if before_health is None:
        before_health = calculate_health_score(dataframe, before_profile)

    cleaned = apply_cleaning_plan(dataframe, plan, copy=True)
    after_profile = build_profile(cleaned)
    after_health = calculate_health_score(cleaned, after_profile)

    before_shape = before_profile.get("shape", {})
    after_shape = after_profile.get("shape", {})
    before_score = before_health.get("overall_score")
    after_score = after_health.get("overall_score")

    health_delta = None
    if isinstance(before_score, (int, float)) and isinstance(after_score, (int, float)):
        health_delta = round(float(after_score) - float(before_score), 4)

    return {
        "before_shape": dict(before_shape),
        "after_shape": dict(after_shape),
        "rows_removed": int(before_shape.get("rows", len(dataframe)))
        - int(after_shape.get("rows", len(cleaned))),
        "missing_before": int(dataframe.isna().sum().sum()),
        "missing_after": int(cleaned.isna().sum().sum()),
        "duplicates_before": int(before_profile.get("duplicate_rows", 0)),
        "duplicates_after": int(after_profile.get("duplicate_rows", 0)),
        "before_health": before_health,
        "after_health": after_health,
        "health_delta": health_delta,
    }
