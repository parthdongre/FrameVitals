import json

import pandas as pd
import pytest

import framevitals
from framevitals.contracts import infer_contract, validate_contract


def _reference_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "account_id": [101, 102, 103],
        "balance": [10.5, 25.0, 40.0],
        "plan": ["basic", "standard", "premium"],
        "joined_at": pd.to_datetime([
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]),
    })


def test_infer_contract_v2_is_json_safe_and_uses_tolerant_bounds():
    contract = infer_contract(_reference_dataframe())

    assert contract["version"] == 2
    assert contract["allow_extra_columns"] is False
    assert contract["extra_columns_severity"] == "error"
    assert contract["inference"]["numeric_tolerance"] == 0.05

    account = contract["columns"]["account_id"]
    balance = contract["columns"]["balance"]
    plan = contract["columns"]["plan"]

    assert account["minimum"] < 101
    assert account["maximum"] > 103
    assert balance["minimum"] < 10.5
    assert balance["maximum"] > 40.0
    assert plan["allowed_values"] == ["basic", "premium", "standard"]
    assert plan["allowed_values_severity"] == "warning"
    assert json.loads(json.dumps(contract)) == contract


def test_inferred_contract_accepts_normal_future_values_inside_tolerance():
    contract = infer_contract(_reference_dataframe(), numeric_tolerance=0.10)
    candidate = pd.DataFrame({
        "account_id": [100, 104],
        "balance": [9.0, 42.0],
        "plan": ["basic", "premium"],
        "joined_at": pd.to_datetime(["2026-01-04", "2026-01-05"]),
    })

    result = validate_contract(candidate, contract)

    assert result["valid"] is True
    assert result["status"] == "pass"
    assert result["summary"]["errors"] == 0


def test_allowed_value_drift_is_warning_not_failure_by_default():
    contract = infer_contract(_reference_dataframe())
    candidate = _reference_dataframe().copy()
    candidate.loc[0, "plan"] = "enterprise"

    result = validate_contract(candidate, contract)

    assert result["valid"] is True
    assert result["status"] == "warn"
    assert result["errors"] == []
    assert result["warnings"][0]["code"] == "allowed_values_violation"
    assert result["warnings"][0]["severity"] == "warning"


def test_inferred_uniqueness_requires_enough_rows_and_detects_duplicates():
    reference = pd.DataFrame({
        "customer_id": [f"CUST-{index:03d}" for index in range(30)],
        "value": list(range(30)),
    })
    contract = infer_contract(reference)

    assert contract["columns"]["customer_id"]["unique"] is True

    candidate = reference.copy()
    candidate.loc[1, "customer_id"] = candidate.loc[0, "customer_id"]
    result = validate_contract(candidate, contract)

    assert result["valid"] is False
    assert any(item["code"] == "uniqueness_violation" for item in result["errors"])


def test_nullable_column_can_warn_when_null_rate_exceeds_expected_ceiling():
    contract = {
        "version": 2,
        "columns": {
            "value": {
                "type": "number",
                "required": True,
                "nullable": True,
                "max_null_fraction": 0.10,
            },
        },
    }
    candidate = pd.DataFrame({"value": [1.0, None, None, 4.0]})

    result = validate_contract(candidate, contract)

    assert result["valid"] is True
    assert result["status"] == "warn"
    assert result["warnings"][0]["code"] == "null_fraction_violation"


def test_optional_contract_column_may_be_absent():
    contract = {
        "version": 2,
        "columns": {
            "required_value": {"type": "integer", "nullable": False},
            "optional_note": {
                "type": "string",
                "required": False,
                "nullable": True,
            },
        },
    }
    candidate = pd.DataFrame({"required_value": [1, 2, 3]})

    result = validate_contract(candidate, contract)

    assert result["valid"] is True
    assert result["status"] == "pass"


def test_validation_aggregates_schema_type_nullability_and_bound_errors():
    contract = infer_contract(_reference_dataframe(), numeric_tolerance=0)
    candidate = pd.DataFrame({
        "balance": [5.0, 55.0],
        "plan": [1, 2],
        "joined_at": pd.to_datetime(["2026-01-04", "2026-01-05"]),
        "unexpected": ["a", "b"],
    })

    result = validate_contract(candidate, contract)
    codes = {finding["code"] for finding in result["errors"]}

    assert result["valid"] is False
    assert result["status"] == "fail"
    assert {
        "missing_column",
        "unexpected_column",
        "incompatible_type",
        "minimum_violation",
        "maximum_violation",
    }.issubset(codes)
    assert result["summary"]["findings"] == len(result["findings"])
    assert result["summary"]["failed_columns"]
    assert result["summary"]["code_counts"]


def test_version_1_contracts_remain_supported():
    contract = {
        "version": 1,
        "columns": {
            "age": {
                "type": "integer",
                "nullable": False,
                "minimum": 18,
                "maximum": 100,
            },
        },
    }

    result = validate_contract(pd.DataFrame({"age": [15, 25]}), contract)

    assert result["contract_version"] == 1
    assert result["status"] == "fail"
    assert any(item["code"] == "minimum_violation" for item in result["errors"])


def test_extra_columns_can_be_allowed_or_downgraded_to_warning():
    candidate = pd.DataFrame({"age": [20], "new_col": [1]})

    allowed = validate_contract(
        candidate,
        {
            "version": 2,
            "allow_extra_columns": True,
            "columns": {"age": {"type": "integer", "nullable": False}},
        },
    )
    warning = validate_contract(
        candidate,
        {
            "version": 2,
            "extra_columns_severity": "warning",
            "columns": {"age": {"type": "integer", "nullable": False}},
        },
    )

    assert allowed["status"] == "pass"
    assert warning["status"] == "warn"
    assert warning["warnings"][0]["code"] == "unexpected_column"


def test_validate_contract_rejects_invalid_contract_definitions():
    with pytest.raises(ValueError, match="non-empty 'columns'"):
        validate_contract(_reference_dataframe(), {"columns": {}})

    with pytest.raises(ValueError, match="minimum exceeds maximum"):
        validate_contract(
            _reference_dataframe(),
            {
                "columns": {
                    "balance": {
                        "type": "number",
                        "minimum": 20,
                        "maximum": 10,
                    },
                },
            },
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_contract(
            _reference_dataframe(),
            {
                "version": 2,
                "columns": {
                    "balance": {
                        "type": "number",
                        "max_null_fraction": 1.5,
                    },
                },
            },
        )


def test_public_contract_api_accepts_file_paths_and_inference_controls(tmp_path):
    reference_path = tmp_path / "reference.csv"
    candidate_path = tmp_path / "candidate.csv"
    _reference_dataframe().to_csv(reference_path, index=False)
    _reference_dataframe().to_csv(candidate_path, index=False)

    contract = framevitals.infer_contract(
        reference_path,
        numeric_tolerance=0.10,
        max_categories=10,
        allow_extra_columns=True,
    )
    result = framevitals.validate(candidate_path, contract)

    assert contract["reference_name"] == "reference.csv"
    assert contract["inference"]["numeric_tolerance"] == 0.10
    assert contract["allow_extra_columns"] is True
    assert result["dataset_name"] == "candidate.csv"
    assert result["valid"] is True
