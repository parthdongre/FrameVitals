"""Extensible user-defined checks for FrameVitals quality gates."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from framevitals.provenance import execution_provenance, load_fully_materializes
from framevitals.quality_results import CheckResult
from framevitals.sources import resolve_source


CheckSeverity = Literal["warning", "error"]
CheckFunction = Callable[[pd.DataFrame], bool | np.bool_ | Mapping[str, Any]]
DataInput = Any


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "check"


def _validate_severity(value: str) -> CheckSeverity:
    if value not in {"warning", "error"}:
        raise ValueError("check severity must be 'warning' or 'error'.")
    return cast(CheckSeverity, value)


@dataclass(frozen=True, slots=True)
class DataCheck:
    """Named user-defined data check with gate severity metadata."""

    name: str
    function: CheckFunction
    severity: CheckSeverity = "error"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("check name must not be empty.")
        if not callable(self.function):
            raise TypeError("check function must be callable.")
        _validate_severity(self.severity)

    def __call__(self, dataframe: pd.DataFrame) -> bool | np.bool_ | Mapping[str, Any]:
        return self.function(dataframe)


def check(
    name: str | None = None,
    *,
    severity: CheckSeverity = "error",
    description: str | None = None,
):
    """Decorate a DataFrame predicate as a reusable :class:`DataCheck`.

    A check can return a boolean or a mapping containing at least ``passed``.
    Optional mapping keys include ``message`` and ``details``.
    """
    resolved_severity = _validate_severity(severity)

    def decorator(function: CheckFunction) -> DataCheck:
        resolved_name = (name or getattr(function, "__name__", "check")).strip()
        return DataCheck(
            name=resolved_name,
            function=function,
            severity=resolved_severity,
            description=description,
        )

    return decorator


def _normalize_check(value: DataCheck | CheckFunction) -> DataCheck:
    if isinstance(value, DataCheck):
        return value
    if not callable(value):
        raise TypeError("custom checks must be DataCheck instances or callables.")
    return DataCheck(
        name=getattr(value, "__name__", "check"),
        function=value,
    )


def _normalize_outcome(
    definition: DataCheck,
    raw: bool | np.bool_ | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(raw, (bool, np.bool_)):
        passed = bool(raw)
        message = (
            f"{definition.name} passed."
            if passed
            else f"{definition.name} failed."
        )
        details: Any = None
    elif isinstance(raw, Mapping):
        if "passed" not in raw:
            raise ValueError(
                f"Custom check {definition.name!r} returned a mapping without 'passed'."
            )
        passed = bool(raw["passed"])
        default_message = (
            f"{definition.name} passed."
            if passed
            else f"{definition.name} failed."
        )
        message = str(raw.get("message") or default_message)
        details = raw.get("details")
    else:
        raise TypeError(
            f"Custom check {definition.name!r} must return bool or a mapping."
        )

    return {
        "name": definition.name,
        "code": f"custom.{_slug(definition.name)}",
        "passed": passed,
        "severity": definition.severity,
        "description": definition.description,
        "message": message,
        "details": details,
    }


def run_checks(
    data: DataInput,
    checks: Sequence[DataCheck | CheckFunction],
) -> CheckResult:
    """Run user-defined checks exactly against a materialized DataFrame.

    Custom Python callables can inspect arbitrary row-level relationships, so
    FrameVitals does not silently sample their input. Non-pandas sources are
    materialized intentionally and that decision is disclosed in ``execution``.
    """
    definitions = [_normalize_check(value) for value in checks]
    if not definitions:
        raise ValueError("run_checks requires at least one custom check.")

    source = resolve_source(data)
    metadata = source.inspect()
    dataframe = source.load()

    results: list[dict[str, Any]] = []
    for definition in definitions:
        try:
            raw = definition(dataframe.copy())
            outcome = _normalize_outcome(definition, raw)
            outcome["execution_error"] = None
        except Exception as exc:  # user-defined check failures become structured results
            outcome = {
                "name": definition.name,
                "code": f"custom.{_slug(definition.name)}",
                "passed": False,
                "severity": "error",
                "description": definition.description,
                "message": f"Custom check raised {type(exc).__name__}: {exc}",
                "details": None,
                "execution_error": f"{type(exc).__name__}: {exc}",
            }
        results.append(outcome)

    failures = [result for result in results if not result["passed"]]
    error_failures = [
        result for result in failures if result.get("severity") == "error"
    ]
    warning_failures = [
        result for result in failures if result.get("severity") == "warning"
    ]
    status = "fail" if error_failures else "warn" if warning_failures else "pass"

    findings = [
        {
            "code": result["code"],
            "severity": result["severity"],
            "title": result["name"],
            "message": result["message"],
            "details": result.get("details"),
        }
        for result in failures
    ]

    execution = execution_provenance(
        "exact_custom_checks",
        full_materialization=load_fully_materializes(metadata),
        source=metadata.to_dict(),
        sampled=False,
        source_rows=metadata.rows,
        source_columns=metadata.columns,
        reason=(
            "Arbitrary custom Python checks run on the complete DataFrame; "
            "FrameVitals does not silently sample user-defined invariants."
        ),
    )

    return CheckResult({
        "status": status,
        "passed": status != "fail",
        "results": results,
        "findings": findings,
        "summary": {
            "checks": len(results),
            "passed": len(results) - len(failures),
            "warnings": len(warning_failures),
            "errors": len(error_failures),
        },
        "execution": execution,
    })
