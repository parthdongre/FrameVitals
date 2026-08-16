import pandas as pd
import pytest

import framevitals as fv


def _reference() -> pd.DataFrame:
    return pd.DataFrame({
        "age": list(range(20, 50)),
        "plan": ["basic", "pro", "team"] * 10,
    })


def test_gate_requires_at_least_one_check():
    with pytest.raises(ValueError, match="reference=.*contract"):
        fv.gate(_reference())


def test_validation_only_gate_passes_clean_data():
    reference = _reference()
    contract = fv.infer_contract(reference)

    result = fv.gate(reference.copy(), contract=contract)

    assert isinstance(result, fv.GateResult)
    assert isinstance(result, dict)
    assert result.status == "pass"
    assert result.passed is True
    assert result["checks_run"] == ["validation"]
    assert result["checks"]["validation"]["status"] == "pass"
    assert "FrameVitals quality gate" in result.summary_text()


def test_validation_warning_can_warn_or_be_promoted_to_failure():
    reference = _reference()
    contract = fv.infer_contract(reference)
    current = reference.copy()
    current.loc[0, "plan"] = "enterprise"

    warning = fv.gate(current, contract=contract)
    strict = fv.gate(
        current,
        contract=contract,
        fail_on_validation_warning=True,
    )

    assert warning.status == "warn"
    assert warning.passed is True
    assert strict.status == "fail"
    assert strict.passed is False
    assert any("promoted to failure" in reason for reason in strict.reasons)


def test_drift_only_gate_uses_configurable_thresholds():
    reference = pd.DataFrame({"value": list(range(50))})
    current = pd.DataFrame({"value": list(range(100, 150))})

    default = fv.gate(current, reference=reference)
    strict = fv.gate(
        current,
        reference=reference,
        drift_warn_on="minor",
        drift_fail_on="moderate",
    )

    assert default.status in {"warn", "fail"}
    assert strict.status == "fail"
    assert strict.passed is False
    assert strict["checks_run"] == ["drift"]
    assert strict["checks"]["drift"]["available"] is True


def test_combined_gate_failure_wins_over_warning():
    reference = _reference()
    contract = fv.infer_contract(reference, numeric_tolerance=0)
    current = reference.copy()
    current["age"] = current["age"] + 100
    current.loc[0, "plan"] = "enterprise"

    result = fv.gate(
        current,
        reference=reference,
        contract=contract,
        drift_warn_on="minor",
        drift_fail_on="moderate",
    )

    assert result.status == "fail"
    assert result.passed is False
    assert set(result["checks_run"]) == {"validation", "drift"}
    assert result["checks"]["validation"]["status"] in {"warn", "fail"}
    assert result["checks"]["drift"]["available"] is True
    assert result.reasons


def test_requested_but_unavailable_drift_fails_gate():
    reference = pd.DataFrame({"left": list(range(30))})
    current = pd.DataFrame({"right": list(range(30))})

    result = fv.gate(current, reference=reference)

    assert result.status == "fail"
    assert result.passed is False
    assert result["checks"]["drift"]["available"] is False
    assert any("could not produce" in reason for reason in result.reasons)


def test_gate_validates_threshold_order_and_max_columns():
    frame = _reference()

    with pytest.raises(ValueError, match="drift_warn_on"):
        fv.gate(frame, reference=frame, drift_warn_on="extreme")
    with pytest.raises(ValueError, match="drift_fail_on"):
        fv.gate(frame, reference=frame, drift_fail_on="extreme")
    with pytest.raises(ValueError, match="cannot be more severe"):
        fv.gate(
            frame,
            reference=frame,
            drift_warn_on="severe",
            drift_fail_on="moderate",
        )
    with pytest.raises(ValueError, match="max_columns"):
        fv.gate(frame, reference=frame, max_columns=0)
