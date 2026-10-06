"""Branded workflow protocols for the public FrameVitals experience.

Protocols are intentionally thin orchestration layers over the stable lower-level
APIs. They reduce the number of decisions users need to make without creating a
second analysis engine or hiding reproducibility from developers.

Technical composition is documented in docs/protocols.md. Product-facing
surfaces should prefer protocol names and outcomes over implementation details.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from framevitals.quality_results import DriftResult, GateResult, ValidationResult
from framevitals.result import AnalysisResult
from framevitals.snapshots import AnalysisSnapshot


class _ProtocolResult(dict):
    """Small dict-compatible base for protocol results."""

    protocol = "protocol"

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

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


class PrismResult(_ProtocolResult):
    """Unified result returned by framevitals.prism."""

    protocol = "prism"

    @property
    def analysis(self) -> AnalysisResult:
        value = self.get("analysis")
        if isinstance(value, AnalysisResult):
            return value
        if isinstance(value, Mapping):
            return AnalysisResult(dict(value))
        return AnalysisResult({})

    @property
    def verdict(self) -> GateResult | None:
        value = self.get("verdict")
        if isinstance(value, GateResult):
            return value
        if isinstance(value, Mapping):
            return GateResult(dict(value))
        return None

    @property
    def change(self) -> DriftResult | None:
        value = self.get("change")
        if isinstance(value, DriftResult):
            return value
        if isinstance(value, Mapping):
            return DriftResult(dict(value))
        return None

    @property
    def tide(self) -> DriftResult | None:
        """Compatibility alias for the change outcome."""
        return self.change

    @property
    def validation(self) -> ValidationResult | None:
        trust = self.get("trust")
        if not isinstance(trust, Mapping):
            return None
        value = trust.get("validation")
        if isinstance(value, ValidationResult):
            return value
        if isinstance(value, Mapping):
            return ValidationResult(dict(value))
        return None

    @property
    def trust(self) -> ValidationResult | None:
        return self.validation

    @property
    def contract(self) -> dict[str, Any] | None:
        trust = self.get("trust")
        if not isinstance(trust, Mapping):
            return None
        value = trust.get("expectations")
        return dict(value) if isinstance(value, Mapping) else None

    @property
    def beacons(self) -> list[dict[str, Any]]:
        """High-priority signals surfaced by Prism."""
        return self.analysis.findings

    @property
    def status(self) -> str:
        verdict = self.verdict
        if verdict is not None:
            return verdict.status
        return str(self.get("status", "complete"))

    def to_public_dict(self) -> dict[str, Any]:
        """Return the product-facing Prism payload without orchestration details."""
        analysis_summary = self.analysis.summary()
        validation = self.validation
        change = self.change
        verdict = self.verdict

        trust_summary = None
        if validation is not None:
            details = validation.get("summary", {})
            trust_summary = {
                "status": validation.status,
                "errors": details.get("errors", 0),
                "warnings": details.get("warnings", 0),
            }

        change_summary = None
        if change is not None:
            details = change.get("summary", {})
            change_summary = {
                "status": change.status,
                "severity": change.severity,
                "columns_compared": details.get("n_columns_compared", 0),
            }

        verdict_summary = None
        if verdict is not None:
            verdict_summary = {
                "status": verdict.status,
                "passed": verdict.passed,
                "reasons": verdict.reasons,
            }

        return {
            "protocol": "prism",
            "status": self.status,
            "summary": analysis_summary,
            "beacons": self.beacons,
            "trust": trust_summary,
            "change": change_summary,
            "verdict": verdict_summary,
        }

    def to_json(
        self,
        destination: str | Path | None = None,
        *,
        indent: int = 2,
    ) -> str | Path:
        rendered = json.dumps(self.to_public_dict(), indent=indent, default=str)
        if destination is None:
            return rendered
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        return path

    def summary_text(self) -> str:
        analysis = self.analysis
        summary = analysis.summary()
        shape = summary.get("shape", {})
        health = summary.get("health", {})
        ml = summary.get("ml_readiness", {})

        source_kind = str(analysis.get("source_kind") or "tabular")
        lines = [
            "FrameVitals · Prism",
            "=" * 72,
            f"Source        {summary.get('filename') or '<unknown>'}",
            f"Kind          {source_kind.upper()}",
            f"Status        {self.status.upper()}",
        ]

        if source_kind == "graph":
            graph = analysis.get("graph", {})
            lines.extend([
                f"Nodes         {graph.get('nodes', '?')}",
                f"Edges         {graph.get('edges', '?')}",
            ])
        elif source_kind == "model":
            model = analysis.get("model", {})
            lines.extend([
                f"Architecture  {model.get('architecture', 'unknown')}",
                f"Parameters    {model.get('parameters', '?')}",
                f"Trainable     {model.get('trainable_parameters', '?')}",
            ])
        elif source_kind == "tensor":
            tensor = analysis.get("tensor", {})
            lines.extend([
                f"Shape         {tensor.get('shape', '?')}",
                f"Dtype         {tensor.get('dtype', '?')}",
                f"Values        {tensor.get('size', '?')}",
            ])
        elif source_kind == "nested":
            nested = analysis.get("nested", {})
            lines.extend([
                f"Nodes         {nested.get('nodes_observed', '?')}",
                f"Depth         {nested.get('max_depth', '?')}",
                f"Type conflicts {nested.get('path_type_conflict_count', '?')}",
            ])
        elif source_kind == "relational":
            relational = analysis.get("relational", {})
            lines.extend([
                f"Tables        {relational.get('table_count', '?')}",
                f"Relationships {relational.get('relationship_count', '?')}",
                f"Isolated      {len(relational.get('isolated_tables', []))}",
            ])
        else:
            lines.extend([
                (
                    "Shape         "
                    f"{shape.get('rows', '?')} rows x {shape.get('columns', '?')} columns"
                ),
                f"ML readiness  {ml.get('score', 'n/a')}  {ml.get('label', '')}",
            ])

        lines.extend([
            f"Health        {health.get('overall_score', 'n/a')}  {health.get('label', '')}",
            f"Beacons       {len(self.beacons)}",
        ])

        validation = self.validation
        if validation is not None:
            lines.append(f"Reference trust {validation.status.upper()}")

        tide_result = self.tide
        if tide_result is not None:
            lines.append(f"Change        {tide_result.severity.upper()}")

        lines.extend([
            "=" * 72,
            "Prism complete.",
        ])
        return "\n".join(lines)

    def to_html(self, destination: str | Path | None = None) -> str | Path:
        return self.analysis.to_html(destination)


class AxiomResult(_ProtocolResult):
    """Expectation set and optional validation outcome."""

    protocol = "axiom"

    @property
    def contract(self) -> dict[str, Any]:
        value = self.get("contract", {})
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def validation(self) -> ValidationResult | None:
        value = self.get("validation")
        if isinstance(value, ValidationResult):
            return value
        if isinstance(value, Mapping):
            return ValidationResult(dict(value))
        return None

    @property
    def status(self) -> str:
        validation = self.validation
        if validation is not None:
            return validation.status
        return str(self.get("status", "established"))

    def summary_text(self) -> str:
        validation = self.validation
        lines = [
            "FrameVitals · Axiom",
            "=" * 72,
            f"Status        {self.status.upper()}",
            f"Reference     {self.contract.get('reference_name', 'dataset')}",
        ]
        if validation is not None:
            summary = validation.get("summary", {})
            lines.extend([
                f"Errors        {summary.get('errors', 0)}",
                f"Warnings      {summary.get('warnings', 0)}",
            ])
        lines.extend(["=" * 72, "Axiom complete."])
        return "\n".join(lines)


class ForgeResult(_ProtocolResult):
    """Cleaning plan with an optional cleaned dataset held as Python metadata."""

    protocol = "forge"

    def __init__(self, *args, cleaned: Any = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cleaned = cleaned

    @property
    def cleaned(self) -> Any:
        return self._cleaned

    @property
    def plan(self) -> dict[str, Any]:
        value = self.get("plan", {})
        return dict(value) if isinstance(value, Mapping) else {}

    @property
    def applied(self) -> bool:
        return bool(self.get("applied", False))

    def summary_text(self) -> str:
        return "\n".join([
            "FrameVitals · Forge",
            "=" * 72,
            f"Status        {'APPLIED' if self.applied else 'READY'}",
            "=" * 72,
            "Forge complete.",
        ])


def prism(
    data: Any,
    *,
    reference: Any = None,
    axiom: Mapping[str, Any] | None = None,
    focus: str | None = None,
    depth: str | None = None,
    contract: Mapping[str, Any] | None = None,
    target: str | None = None,
    mode: str | None = None,
    artifacts: bool | None = None,
    workers: int | None = None,
    preset: str | None = None,
    config: Any = None,
    disabled_modules: list[str] | tuple[str, ...] | None = None,
    max_sample_rows: int | None = None,
    max_relationship_pairs: int | None = None,
    max_memory_heavy_parallelism: int | None = None,
    max_streaming_profile_columns: int | None = None,
    sample_batch: Any = None,
    targets: Any = None,
    loss_fn: Any = None,
    backward: bool = False,
    max_runtime_modules: int | None = None,
    optimizer: Any = None,
    derive_axiom: bool = True,
    custom_checks: Sequence[Any] | None = None,
    columns: list[str] | None = None,
    max_columns: int = 30,
    drift_warn_on: str = "moderate",
    drift_fail_on: str = "severe",
    fail_on_validation_warning: bool = False,
) -> PrismResult:
    """Run the comprehensive FrameVitals protocol.

    With only data, Prism performs the canonical analysis. When a reference
    or explicit contract is supplied, Prism also coordinates the corresponding
    trust checks and returns one coherent protocol result.
    """
    from framevitals.analysis_api import analyze
    from framevitals.operations import gate, infer_contract

    if axiom is not None and contract is not None:
        raise ValueError("Pass either axiom= or contract=, not both.")
    if focus is not None and target is not None and focus != target:
        raise ValueError("focus= and target= cannot disagree.")
    if depth is not None and mode is not None and depth != mode:
        raise ValueError("depth= and mode= cannot disagree.")

    resolved_focus = focus if focus is not None else target
    resolved_depth = depth if depth is not None else mode
    supplied_expectations = axiom if axiom is not None else contract

    # Structured non-tabular sources are recognized before entering the mature
    # tabular dispatcher. This keeps graph/tensor/model diagnostics isolated
    # from pandas-specific execution while preserving one public Prism call.
    from framevitals.structured_analysis import analyze_structured

    structured_analysis = analyze_structured(
        data,
        depth=resolved_depth,
        sample_batch=sample_batch,
        targets=targets,
        loss_fn=loss_fn,
        backward=backward,
        max_runtime_modules=max_runtime_modules,
        optimizer=optimizer,
    )
    if structured_analysis is not None:
        if resolved_focus is not None:
            raise ValueError("focus=/target= is currently supported only for tabular Prism input.")
        if custom_checks:
            raise NotImplementedError(
                "custom_checks= is currently available only for tabular Prism input."
            )

        from framevitals.analysis.structured_contracts import (
            infer_structured_contract,
            validate_structured,
        )
        from framevitals.structured_analysis import compare_structured

        resolved_contract = (
            dict(supplied_expectations)
            if isinstance(supplied_expectations, Mapping)
            else None
        )
        if reference is not None and resolved_contract is None and derive_axiom:
            resolved_contract = infer_structured_contract(reference)

        validation_result: ValidationResult | None = None
        if resolved_contract is not None:
            validation_result = validate_structured(data, resolved_contract)

        tide_result: DriftResult | None = None
        if reference is not None:
            compared = compare_structured(reference, data)
            if isinstance(compared, DriftResult):
                tide_result = compared
            elif isinstance(compared, Mapping):
                tide_result = DriftResult(dict(compared))

        statuses = [
            value
            for value in (
                validation_result.status if validation_result is not None else None,
                tide_result.status if tide_result is not None else None,
            )
            if value
        ]
        if "fail" in statuses:
            structured_status = "fail"
        elif "warn" in statuses:
            structured_status = "warn"
        elif statuses:
            structured_status = "pass"
        else:
            structured_status = "complete"

        trust_payload = None
        if resolved_contract is not None:
            trust_payload = {
                "expectations": resolved_contract,
                "validation": validation_result,
            }

        return PrismResult({
            "protocol": "prism",
            "status": structured_status,
            "analysis": structured_analysis,
            "trust": trust_payload,
            "change": tide_result,
            "verdict": None,
        })

    if (
        sample_batch is not None
        or targets is not None
        or loss_fn is not None
        or backward
        or max_runtime_modules is not None
        or optimizer is not None
    ):
        raise ValueError(
            "sample_batch=/targets=/loss_fn=/backward= are currently "
            "supported only for model Prism input."
        )

    analysis = analyze(
        data,
        target=resolved_focus,
        mode=resolved_depth,
        artifacts=artifacts,
        workers=workers,
        preset=preset,
        config=config,
        disabled_modules=disabled_modules,
        max_sample_rows=max_sample_rows,
        max_relationship_pairs=max_relationship_pairs,
        max_memory_heavy_parallelism=max_memory_heavy_parallelism,
        max_streaming_profile_columns=max_streaming_profile_columns,
    )

    resolved_contract = (
        dict(supplied_expectations)
        if isinstance(supplied_expectations, Mapping)
        else None
    )
    if reference is not None and resolved_contract is None and derive_axiom:
        resolved_contract = infer_contract(reference)

    verdict: GateResult | None = None
    tide_result: DriftResult | None = None
    validation_result: ValidationResult | None = None

    if reference is not None or resolved_contract is not None or custom_checks:
        verdict = gate(
            data,
            reference=reference,
            contract=resolved_contract,
            custom_checks=custom_checks,
            columns=columns,
            max_columns=max_columns,
            drift_warn_on=drift_warn_on,
            drift_fail_on=drift_fail_on,
            fail_on_validation_warning=fail_on_validation_warning,
        )
        checks = verdict.get("checks", {})
        if isinstance(checks, Mapping):
            drift_payload = checks.get("drift")
            if isinstance(drift_payload, DriftResult):
                tide_result = drift_payload
            elif isinstance(drift_payload, Mapping):
                tide_result = DriftResult(dict(drift_payload))

            validation_payload = checks.get("validation")
            if isinstance(validation_payload, ValidationResult):
                validation_result = validation_payload
            elif isinstance(validation_payload, Mapping):
                validation_result = ValidationResult(dict(validation_payload))

    trust_payload = None
    if resolved_contract is not None:
        trust_payload = {
            "expectations": resolved_contract,
            "validation": validation_result,
        }

    return PrismResult({
        "protocol": "prism",
        "status": verdict.status if verdict is not None else "complete",
        "analysis": analysis,
        "trust": trust_payload,
        "change": tide_result,
        "verdict": verdict,
    })


def axiom(
    reference: Any,
    *,
    current: Any = None,
    contract: Mapping[str, Any] | None = None,
    numeric_tolerance: float = 0.05,
    max_categories: int = 20,
    null_fraction_tolerance: float = 0.05,
    infer_unique: bool = True,
    min_unique_rows: int = 20,
    allow_extra_columns: bool = False,
) -> AxiomResult:
    """Establish expectations from a reference and optionally test current data."""
    from framevitals.core.source import SourceKind, recognize_source

    descriptor = recognize_source(reference)
    structured_kinds = {
        SourceKind.GRAPH,
        SourceKind.TENSOR,
        SourceKind.NESTED,
        SourceKind.RELATIONAL,
        SourceKind.MODEL,
    }

    if descriptor.kind in structured_kinds:
        from framevitals.analysis.structured_contracts import (
            infer_structured_contract,
            validate_structured,
        )

        resolved = (
            dict(contract)
            if isinstance(contract, Mapping)
            else infer_structured_contract(
                reference,
                tolerance=max(float(numeric_tolerance), 0.0),
            )
        )
        if resolved is None:
            raise TypeError(
                f"Axiom does not support source kind {descriptor.kind.value!r}."
            )
        validation = (
            validate_structured(current, resolved)
            if current is not None
            else None
        )
        return AxiomResult({
            "protocol": "axiom",
            "status": validation.status if validation is not None else "established",
            "contract": resolved,
            "validation": validation,
        })

    from framevitals.operations import infer_contract, validate

    resolved = (
        dict(contract)
        if isinstance(contract, Mapping)
        else infer_contract(
            reference,
            numeric_tolerance=numeric_tolerance,
            max_categories=max_categories,
            null_fraction_tolerance=null_fraction_tolerance,
            infer_unique=infer_unique,
            min_unique_rows=min_unique_rows,
            allow_extra_columns=allow_extra_columns,
        )
    )
    validation = validate(current, resolved) if current is not None else None

    return AxiomResult({
        "protocol": "axiom",
        "status": validation.status if validation is not None else "established",
        "contract": resolved,
        "validation": validation,
    })


def tide(
    reference: Any,
    current: Any,
    *,
    columns: list[str] | None = None,
    max_columns: int = 30,
) -> DriftResult:
    """Run the FrameVitals change protocol across two source states."""
    from framevitals.structured_analysis import compare_structured

    structured = compare_structured(reference, current)
    if structured is not None:
        if columns is not None:
            raise ValueError("columns= is only valid for tabular Tide comparisons.")
        return structured

    from framevitals.operations import compare

    return compare(reference, current, columns=columns, max_columns=max_columns)


def forge(
    data: Any,
    *,
    apply: bool = False,
    plan: Mapping[str, Any] | None = None,
) -> ForgeResult:
    """Prepare a conservative transformation plan and optionally apply it."""
    from framevitals.operations import clean, plan_cleaning

    resolved_plan = plan_cleaning(data) if plan is None else plan
    summary = (
        resolved_plan.summary()
        if hasattr(resolved_plan, "summary")
        else {"plan": dict(resolved_plan)}
    )
    cleaned = clean(data, plan=resolved_plan) if apply else None
    return ForgeResult(
        {
            "protocol": "forge",
            "status": "applied" if apply else "ready",
            "applied": bool(apply),
            "plan": dict(resolved_plan),
            "summary": summary,
        },
        cleaned=cleaned,
    )


def pulse(
    data_or_result: Any,
    *,
    destination: str | Path | None = None,
    depth: str | None = None,
    mode: str | None = None,
    workers: int | None = None,
    sample_batch: Any = None,
    targets: Any = None,
    loss_fn: Any = None,
    backward: bool = False,
    max_runtime_modules: int | None = None,
    optimizer: Any = None,
) -> AnalysisSnapshot:
    """Capture a compact health state from tabular or structured sources."""
    from framevitals.analysis_api import analyze
    from framevitals.snapshots import create_snapshot
    from framevitals.structured_analysis import analyze_structured

    if depth is not None and mode is not None and depth != mode:
        raise ValueError("depth= and mode= cannot disagree.")
    resolved_depth = depth if depth is not None else mode or "quick"

    if isinstance(data_or_result, PrismResult):
        analysis = data_or_result.analysis
    elif isinstance(data_or_result, AnalysisResult):
        analysis = data_or_result
    elif (
        isinstance(data_or_result, Mapping)
        and "result_schema_version" in data_or_result
    ):
        analysis = AnalysisResult(dict(data_or_result))
    else:
        structured = analyze_structured(
            data_or_result,
            depth=resolved_depth,
            sample_batch=sample_batch,
            targets=targets,
            loss_fn=loss_fn,
            backward=backward,
            max_runtime_modules=max_runtime_modules,
            optimizer=optimizer,
        )
        if structured is not None:
            analysis = structured
        else:
            if (
                sample_batch is not None
                or targets is not None
                or loss_fn is not None
                or backward
                or max_runtime_modules is not None
                or optimizer is not None
            ):
                raise ValueError(
                    "Runtime model options are only valid for model Pulse input."
                )
            analysis = analyze(
                data_or_result,
                mode=resolved_depth,
                artifacts=False,
                workers=workers,
            )

    snapshot = create_snapshot(analysis)
    if destination is not None:
        snapshot.to_json(destination)
    return snapshot
