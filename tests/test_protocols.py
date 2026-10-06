import pandas as pd

import framevitals as fv
from framevitals.quality_results import DriftResult, GateResult, ValidationResult
from framevitals.result import AnalysisResult


def _analysis_result() -> AnalysisResult:
    return AnalysisResult(
        {
            "filename": "current.csv",
            "analysis_mode": "standard",
            "profile": {
                "shape": {"rows": 3, "columns": 2},
            },
            "health": {
                "overall_score": 91.0,
                "label": "healthy",
            },
            "ml_readiness": {
                "score": 84.0,
                "label": "ready",
            },
            "findings": [
                {
                    "code": "quality.example",
                    "severity": "medium",
                    "title": "Example beacon",
                }
            ],
        }
    )


def test_prism_is_one_orchestrated_workflow(monkeypatch):
    analysis = _analysis_result()
    contract = {
        "version": 1,
        "reference_name": "training.csv",
        "columns": {},
    }
    validation = ValidationResult(
        {
            "status": "pass",
            "valid": True,
            "summary": {"errors": 0, "warnings": 0},
            "findings": [],
        }
    )
    drift = DriftResult(
        {
            "available": True,
            "summary": {
                "overall_verdict": "stable",
                "n_columns_compared": 2,
            },
            "gate": {
                "status": "pass",
                "severity": "stable",
            },
            "columns": [],
        }
    )
    verdict = GateResult(
        {
            "status": "pass",
            "passed": True,
            "checks_run": ["validation", "drift"],
            "checks": {
                "validation": validation,
                "drift": drift,
            },
            "reasons": [],
        }
    )

    calls = {"analyze": 0, "infer_contract": 0, "gate": 0}

    def fake_analyze(data, **kwargs):
        calls["analyze"] += 1
        assert data == "current.csv"
        assert kwargs["target"] == "churn"
        return analysis

    def fake_infer_contract(reference):
        calls["infer_contract"] += 1
        assert reference == "training.csv"
        return contract

    def fake_gate(current, **kwargs):
        calls["gate"] += 1
        assert current == "current.csv"
        assert kwargs["reference"] == "training.csv"
        assert kwargs["contract"] == contract
        return verdict

    monkeypatch.setattr("framevitals.analysis_api.analyze", fake_analyze)
    monkeypatch.setattr("framevitals.operations.infer_contract", fake_infer_contract)
    monkeypatch.setattr("framevitals.operations.gate", fake_gate)

    result = fv.prism(
        "current.csv",
        reference="training.csv",
        focus="churn",
    )

    assert isinstance(result, fv.PrismResult)
    assert result.analysis is analysis
    assert result.contract == contract
    assert result.validation.status == "pass"
    assert result.tide.severity == "stable"
    assert result.verdict.passed is True
    assert result.status == "pass"
    assert result.beacons[0]["title"] == "Example beacon"
    assert calls == {"analyze": 1, "infer_contract": 1, "gate": 1}


def test_prism_without_reference_remains_single_analysis(monkeypatch):
    analysis = _analysis_result()

    monkeypatch.setattr(
        "framevitals.analysis_api.analyze",
        lambda data, **kwargs: analysis,
    )

    result = fv.prism("current.csv")

    assert result.analysis is analysis
    assert result.contract is None
    assert result.validation is None
    assert result.tide is None
    assert result.verdict is None
    assert result.status == "complete"


def test_axiom_establishes_and_optionally_tests_expectations(monkeypatch):
    contract = {
        "version": 1,
        "reference_name": "reference.csv",
        "columns": {"value": {"type": "integer"}},
    }
    validation = ValidationResult(
        {
            "status": "warn",
            "valid": True,
            "summary": {"errors": 0, "warnings": 1},
            "findings": [],
        }
    )

    monkeypatch.setattr(
        "framevitals.operations.infer_contract",
        lambda reference, **kwargs: contract,
    )
    monkeypatch.setattr(
        "framevitals.operations.validate",
        lambda current, resolved: validation,
    )

    result = fv.axiom("reference.csv", current="current.csv")

    assert isinstance(result, fv.AxiomResult)
    assert result.contract == contract
    assert result.validation is validation
    assert result.status == "warn"


def test_forge_keeps_application_explicit(monkeypatch):
    plan = {"operations": [{"kind": "dedupe"}]}
    cleaned = pd.DataFrame({"value": [1, 2]})

    monkeypatch.setattr(
        "framevitals.operations.plan_cleaning",
        lambda data: plan,
    )
    monkeypatch.setattr(
        "framevitals.operations.clean",
        lambda data, plan=None: cleaned,
    )

    preview = fv.forge("dataset.csv")
    applied = fv.forge("dataset.csv", apply=True)

    assert isinstance(preview, fv.ForgeResult)
    assert preview.applied is False
    assert preview.cleaned is None
    assert applied.applied is True
    assert applied.cleaned is cleaned
    assert applied.plan == plan


def test_tide_delegates_to_source_aware_compare(monkeypatch):
    drift = DriftResult(
        {
            "available": True,
            "summary": {"overall_verdict": "minor"},
            "gate": {"status": "warn", "severity": "minor"},
            "columns": [],
        }
    )

    monkeypatch.setattr(
        "framevitals.operations.compare",
        lambda reference, current, **kwargs: drift,
    )

    result = fv.tide("before.csv", "after.csv")

    assert result is drift
    assert result.severity == "minor"


def test_pulse_can_capture_an_existing_analysis(tmp_path):
    destination = tmp_path / "pulse.json"

    snapshot = fv.pulse(_analysis_result(), destination=destination)

    assert isinstance(snapshot, fv.AnalysisSnapshot)
    assert snapshot["source"]["filename"] == "current.csv"
    assert destination.exists()



def test_prism_keeps_target_and_mode_as_compatibility_aliases(monkeypatch):
    analysis = _analysis_result()
    captured = {}

    def fake_analyze(data, **kwargs):
        captured.update(kwargs)
        return analysis

    monkeypatch.setattr("framevitals.analysis_api.analyze", fake_analyze)

    result = fv.prism("current.csv", target="churn", mode="quick")

    assert result.analysis is analysis
    assert captured["target"] == "churn"
    assert captured["mode"] == "quick"
