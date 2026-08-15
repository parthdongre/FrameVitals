import pandas as pd

import framevitals
from framevitals.contracts import load_contract, save_contract


def _reference_frame():
    return pd.DataFrame({
        "customer_id": [f"c{i:03d}" for i in range(30)],
        "age": list(range(20, 50)),
        "segment": ["consumer", "business", "consumer"] * 10,
        "score": [float(i) for i in range(30)],
    })


def test_inferred_contract_validates_reference():
    reference = _reference_frame()

    contract = framevitals.infer_contract(reference)
    result = framevitals.validate(reference, contract)

    assert contract["contract_version"] == 1
    assert result["valid"] is True
    assert result["status"] == "pass"
    assert result["summary"]["fail"] == 0
    assert contract["columns"]["customer_id"]["unique"] is True


def test_missing_required_column_fails_validation():
    reference = _reference_frame()
    contract = framevitals.infer_contract(reference)
    current = reference.drop(columns=["age"])

    result = framevitals.validate(current, contract)

    assert result["valid"] is False
    assert result["status"] == "fail"
    assert "age" in result["schema"]["missing_columns"]
    assert any(
        check["check"] == "required_column"
        and check["column"] == "age"
        and check["status"] == "fail"
        for check in result["checks"]
    )


def test_dtype_change_fails_validation():
    reference = _reference_frame()
    contract = framevitals.infer_contract(reference)
    current = reference.copy()
    current["age"] = [f"age-{value}" for value in current["age"]]

    result = framevitals.validate(current, contract)

    assert result["valid"] is False
    assert any(
        check["check"] == "dtype_family"
        and check["column"] == "age"
        and check["status"] == "fail"
        for check in result["checks"]
    )


def test_missingness_budget_fails_when_exceeded():
    reference = _reference_frame()
    contract = framevitals.infer_contract(reference, missing_tolerance=0.05)
    current = reference.copy()
    current.loc[:9, "score"] = None

    result = framevitals.validate(current, contract)

    assert result["valid"] is False
    assert any(
        check["check"] == "missing_fraction"
        and check["column"] == "score"
        and check["status"] == "fail"
        for check in result["checks"]
    )


def test_new_low_cardinality_category_warns_without_failing():
    reference = _reference_frame()
    contract = framevitals.infer_contract(reference)
    current = reference.copy()
    current.loc[0, "segment"] = "enterprise"

    result = framevitals.validate(current, contract)

    assert result["valid"] is True
    assert result["status"] == "warn"
    assert any(
        check["check"] == "allowed_values"
        and check["column"] == "segment"
        and check["status"] == "warn"
        for check in result["checks"]
    )


def test_id_like_uniqueness_is_enforced():
    reference = _reference_frame()
    contract = framevitals.infer_contract(reference)
    current = reference.copy()
    current.loc[1, "customer_id"] = current.loc[0, "customer_id"]

    result = framevitals.validate(current, contract)

    assert result["valid"] is False
    assert any(
        check["check"] == "unique"
        and check["column"] == "customer_id"
        and check["status"] == "fail"
        for check in result["checks"]
    )


def test_extra_column_warns():
    reference = _reference_frame()
    contract = framevitals.infer_contract(reference)
    current = reference.assign(new_signal=1)

    result = framevitals.validate(current, contract)

    assert result["valid"] is True
    assert result["status"] == "warn"
    assert result["schema"]["extra_columns"] == ["new_signal"]


def test_contract_roundtrip(tmp_path):
    contract = framevitals.infer_contract(_reference_frame())
    path = tmp_path / "contract.json"

    save_contract(contract, path)
    loaded = load_contract(path)

    assert loaded == contract
    assert framevitals.validate(_reference_frame(), path)["valid"] is True
