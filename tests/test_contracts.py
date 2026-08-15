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


def test_infer_contract_captures_json_safe_schema_and_bounds():
    contract = infer_contract(_reference_dataframe())

    assert contract == {
        "version": 1,
        "allow_extra_columns": False,
        "columns": {
            "account_id": {
                "type": "integer",
                "nullable": False,
                "minimum": 101,
                "maximum": 103,
            },
            "balance": {
                "type": "number",
                "nullable": False,
                "minimum": 10.5,
                "maximum": 40.0,
            },
            "plan": {"type": "string", "nullable": False},
            "joined_at": {"type": "datetime", "nullable": False},
        },
    }
    assert json.loads(json.dumps(contract)) == contract


def test_validate_contract_accepts_compatible_data_without_mutating_it():
    contract = infer_contract(_reference_dataframe())
    candidate = pd.DataFrame({
        "account_id": [101, 102],
        "balance": [11, 30],
        "plan": ["basic", "premium"],
        "joined_at": pd.to_datetime(["2026-01-04", "2026-01-05"]),
    })
    original = candidate.copy(deep=True)

    result = validate_contract(candidate, contract)

    assert result["valid"] is True
    assert result["summary"] == {
        "columns_checked": 4,
        "errors": 0,
        "warnings": 0,
    }
    pd.testing.assert_frame_equal(candidate, original)


def test_validate_contract_reports_schema_type_nullability_and_bound_errors():
    contract = infer_contract(_reference_dataframe())
    candidate = pd.DataFrame({
        "balance": [5.0, 55.0],
        "plan": [1, 2],
        "joined_at": pd.to_datetime(["2026-01-04", "2026-01-05"]),
        "unexpected": ["a", "b"],
    })

    result = validate_contract(candidate, contract)

    assert result["valid"] is False
    assert {finding["code"] for finding in result["errors"]} == {
        "missing_column",
        "unexpected_column",
        "incompatible_type",
        "minimum_violation",
        "maximum_violation",
    }


def test_validate_contract_rejects_nulls_when_contract_disallows_them():
    contract = infer_contract(_reference_dataframe())
    candidate = _reference_dataframe()
    candidate.loc[1, "plan"] = None

    result = validate_contract(candidate, contract)

    assert result["valid"] is False
    assert result["errors"] == [{
        "code": "nullability_violation",
        "column": "plan",
        "message": "Column 'plan' does not allow null values (1 found).",
        "null_count": 1,
    }]


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


def test_public_contract_api_accepts_file_paths_and_preserves_source_name(tmp_path):
    reference_path = tmp_path / "reference.csv"
    candidate_path = tmp_path / "candidate.csv"
    _reference_dataframe().to_csv(reference_path, index=False)
    _reference_dataframe().to_csv(candidate_path, index=False)

    contract = framevitals.infer_contract(reference_path)
    result = framevitals.validate(candidate_path, contract)

    assert contract["reference_name"] == "reference.csv"
    assert result["dataset_name"] == "candidate.csv"
    assert result["valid"] is True
