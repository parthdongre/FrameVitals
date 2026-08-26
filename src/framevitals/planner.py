"""Declarative execution planning for FrameVitals.

The planner combines signal-driven analysis selection with stable runtime module
policy so ``plan()`` can explain both *what* analyses are applicable and *which*
execution modules are expected to run, be skipped, or remain conditional.

Version 0.3 centralizes mode policy here first. Materialized and streaming
execution can progressively consume the same planner contract without changing
the public result shape or duplicating policy tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from framevitals.analysis_selector import select_analyses
from framevitals.config import VALID_MODULES


PLANNER_SCHEMA_VERSION = "1"
_ALL_MODES = ("quick", "standard", "deep", "research")
_RUNNABLE_STATUSES = frozenset({"run", "conditional"})


MODE_DISABLED_MODULES: dict[str, frozenset[str]] = {
    "quick": frozenset({
        "deep_statistics",
        "anomaly_detection",
        "time_series",
        "text_profile",
        "modeling",
        "explainability",
    }),
    "standard": frozenset({
        "deep_statistics",
        "text_profile",
        "modeling",
        "explainability",
    }),
    "deep": frozenset({"modeling", "explainability"}),
    "research": frozenset(),
}


@dataclass(frozen=True, slots=True)
class ModuleRule:
    """Declarative runtime rule for one execution module."""

    resource_class: str
    modes: tuple[str, ...] = _ALL_MODES
    depends_on: tuple[str, ...] = ()
    requires_target: bool = False
    requires_all_signals: tuple[str, ...] = ()
    requires_any_signals: tuple[str, ...] = ()
    artifacts_required: bool = False
    conditional_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODULE_RULES: dict[str, ModuleRule] = {
    "quality_diagnostics": ModuleRule(resource_class="bounded_cpu"),
    "deep_statistics": ModuleRule(
        resource_class="memory_heavy",
        requires_all_signals=("has_numeric_columns",),
    ),
    "anomaly_detection": ModuleRule(
        resource_class="memory_heavy",
        requires_all_signals=("has_numeric_columns",),
    ),
    "time_series": ModuleRule(
        resource_class="memory_heavy",
        requires_any_signals=("has_datetime_columns", "has_time_series_structure"),
    ),
    "text_profile": ModuleRule(
        resource_class="bounded_cpu",
        requires_all_signals=("has_long_text_columns",),
    ),
    "target_intelligence": ModuleRule(
        resource_class="bounded_cpu",
        requires_target=True,
    ),
    "modeling": ModuleRule(
        resource_class="memory_heavy",
        requires_target=True,
        depends_on=("target_intelligence",),
    ),
    "explainability": ModuleRule(
        resource_class="memory_heavy",
        requires_target=True,
        depends_on=("modeling",),
        conditional_reason="Runs only when modeling produces an explainable winner.",
    ),
    "cleaning": ModuleRule(resource_class="bounded_cpu"),
    "charts": ModuleRule(
        resource_class="artifact_io",
        modes=("standard", "deep", "research"),
        artifacts_required=True,
    ),
    "ai": ModuleRule(
        resource_class="optional_external",
        conditional_reason="Requires explicit runtime AI opt-in and an available provider.",
    ),
}

if set(MODULE_RULES) != VALID_MODULES:
    missing = sorted(VALID_MODULES - set(MODULE_RULES))
    extra = sorted(set(MODULE_RULES) - VALID_MODULES)
    raise RuntimeError(
        "Planner module rules are out of sync with VALID_MODULES "
        f"(missing={missing}, extra={extra})."
    )

for _module_name, _module_rule in MODULE_RULES.items():
    unknown_dependencies = sorted(set(_module_rule.depends_on) - VALID_MODULES)
    if unknown_dependencies:
        raise RuntimeError(
            f"Planner module {_module_name} has unknown dependencies: "
            + ", ".join(unknown_dependencies)
        )


def effective_disabled_modules(
    mode: str,
    user_disabled: Iterable[str] = (),
) -> tuple[str, ...]:
    """Merge explicit disables with the stable built-in policy for ``mode``."""
    implicit = MODE_DISABLED_MODULES.get(mode)
    if implicit is None:
        raise ValueError(f"Unknown analysis mode: {mode}")
    explicit = {str(name) for name in user_disabled}
    unknown = sorted(explicit - VALID_MODULES)
    if unknown:
        raise ValueError("Unknown disabled module(s): " + ", ".join(unknown))
    return tuple(sorted(explicit | set(implicit)))


def _missing_signal_reason(
    rule: ModuleRule,
    signals: Mapping[str, Any],
) -> str | None:
    missing = [name for name in rule.requires_all_signals if not signals.get(name)]
    if missing:
        return "Requires signal(s): " + ", ".join(missing) + "."

    if rule.requires_any_signals and not any(
        signals.get(name) for name in rule.requires_any_signals
    ):
        return "Requires at least one signal: " + ", ".join(rule.requires_any_signals) + "."
    return None


def _apply_dependency_constraints(
    decisions: dict[str, dict[str, Any]],
) -> None:
    """Block modules whose declared upstream dependencies cannot run."""
    changed = True
    while changed:
        changed = False
        for module in sorted(decisions):
            decision = decisions[module]
            if decision["status"] not in _RUNNABLE_STATUSES:
                continue

            blockers = [
                dependency
                for dependency in MODULE_RULES[module].depends_on
                if decisions[dependency]["status"] not in _RUNNABLE_STATUSES
            ]
            if not blockers:
                continue

            decision["status"] = "not_applicable"
            decision["blocked_by"] = blockers
            rendered = ", ".join(
                f"{dependency} ({decisions[dependency]['status']})"
                for dependency in blockers
            )
            decision["reason"] = f"Blocked by dependency: {rendered}."
            changed = True


def _execution_stages(
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Topologically group runnable modules into dependency-safe stages."""
    pending = {
        module
        for module, decision in decisions.items()
        if decision.get("status") in _RUNNABLE_STATUSES
    }
    scheduled: set[str] = set()
    stages: list[dict[str, Any]] = []

    while pending:
        ready = sorted(
            module
            for module in pending
            if all(
                dependency in scheduled or dependency not in pending
                for dependency in MODULE_RULES[module].depends_on
            )
        )
        if not ready:
            cycle = ", ".join(sorted(pending))
            raise RuntimeError(f"Planner dependency cycle detected among: {cycle}")

        stages.append({
            "stage": len(stages),
            "modules": ready,
            "resource_classes": sorted({
                str(decisions[module]["resource_class"])
                for module in ready
            }),
        })
        scheduled.update(ready)
        pending.difference_update(ready)

    return stages


def plan_execution_modules(
    *,
    signals: Mapping[str, Any],
    analysis_mode: str,
    target_column: str | None,
    disabled_modules: Iterable[str] = (),
    artifacts: bool = False,
) -> dict[str, Any]:
    """Return explainable runtime-module decisions for one planned analysis."""
    if analysis_mode not in MODE_DISABLED_MODULES:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")

    explicit_disabled = {str(name) for name in disabled_modules}
    effective_disabled = set(
        effective_disabled_modules(analysis_mode, explicit_disabled)
    )
    implicit_disabled = effective_disabled - explicit_disabled

    decisions: dict[str, dict[str, Any]] = {}

    for module in sorted(VALID_MODULES):
        rule = MODULE_RULES[module]
        status: str
        reason: str

        if module in explicit_disabled:
            status = "disabled_by_config"
            reason = "Disabled by explicit configuration."
        elif module in implicit_disabled:
            status = "disabled_by_mode"
            reason = f"Disabled by the {analysis_mode} mode policy."
        elif analysis_mode not in rule.modes:
            status = "not_applicable"
            reason = f"Not applicable in {analysis_mode} mode."
        elif rule.requires_target and not target_column:
            status = "not_applicable"
            reason = "Requires an explicit target column."
        elif rule.artifacts_required and not artifacts:
            status = "not_applicable"
            reason = "Requires artifact generation to be enabled."
        else:
            signal_reason = _missing_signal_reason(rule, signals)
            if signal_reason is not None:
                status = "not_applicable"
                reason = signal_reason
            elif rule.conditional_reason is not None:
                status = "conditional"
                reason = rule.conditional_reason
            else:
                status = "run"
                reason = "Applicable under the resolved mode, signals, and configuration."

        decisions[module] = {
            "status": status,
            "reason": reason,
            "resource_class": rule.resource_class,
            "depends_on": list(rule.depends_on),
            "blocked_by": [],
        }

    _apply_dependency_constraints(decisions)

    counts: dict[str, int] = {}
    for decision in decisions.values():
        status = str(decision["status"])
        counts[status] = counts.get(status, 0) + 1

    stages = _execution_stages(decisions)
    runnable_modules = [
        module
        for stage in stages
        for module in stage["modules"]
    ]

    return {
        "explicit_disabled": sorted(explicit_disabled),
        "effective_disabled": sorted(effective_disabled),
        "decisions": decisions,
        "summary": dict(sorted(counts.items())),
        "execution_stages": stages,
        "runnable_modules": runnable_modules,
    }


def build_execution_plan(
    *,
    signals: Mapping[str, Any],
    analysis_mode: str,
    target_column: str | None = None,
    disabled_modules: Iterable[str] = (),
    artifacts: bool = False,
) -> dict[str, Any]:
    """Build the stable planner contract consumed by ``framevitals.plan``."""
    selection = select_analyses(
        signals=signals,
        analysis_mode=analysis_mode,
        target_column=target_column,
    )
    modules = plan_execution_modules(
        signals=signals,
        analysis_mode=analysis_mode,
        target_column=target_column,
        disabled_modules=disabled_modules,
        artifacts=artifacts,
    )
    explicit_disabled = set(modules["explicit_disabled"])

    # Preserve the 0.2/early-0.3 compatibility keys while adding the richer
    # versioned decision model alongside them.
    modules["disabled"] = sorted(explicit_disabled)
    modules["enabled"] = sorted(VALID_MODULES - explicit_disabled)

    selection["planner_schema_version"] = PLANNER_SCHEMA_VERSION
    selection["execution_modules"] = modules
    return selection
