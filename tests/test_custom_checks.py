import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import framevitals as fv


def test_top_level_import_keeps_custom_check_engine_lazy():
    src_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import framevitals; assert 'framevitals.checks' not in sys.modules",
        ],
        cwd=src_root.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_check_decorator_and_exact_check_results():
    frame = pd.DataFrame({"revenue": [10.0, 20.0, 30.0]})

    @fv.check("positive revenue", description="Revenue cannot be negative.")
    def positive_revenue(df):
        passed = bool((df["revenue"] >= 0).all())
        return {
            "passed": passed,
            "message": "Revenue is non-negative." if passed else "Negative revenue found.",
            "details": {"minimum": float(df["revenue"].min())},
        }

    assert positive_revenue.name == "positive revenue"
    result = fv.run_checks(frame, [positive_revenue])

    assert result["status"] == "pass"
    assert result["passed"] is True
    assert result["summary"] == {"checks": 1, "passed": 1, "warnings": 0, "errors": 0}
    assert result["results"][0]["code"] == "custom.positive_revenue"
    assert result["results"][0]["details"] == {"minimum": 10.0}
    assert result["execution"]["method"] == "exact_custom_checks"
    assert result["execution"]["full_materialization"] is False


def test_warning_and_error_checks_produce_distinct_statuses():
    frame = pd.DataFrame({"value": [1, 2, 3]})

    @fv.check("soft expectation", severity="warning")
    def soft_expectation(df):
        return False

    @fv.check("hard expectation", severity="error")
    def hard_expectation(df):
        return {"passed": False, "message": "Hard invariant failed."}

    warning_only = fv.run_checks(frame, [soft_expectation])
    assert warning_only["status"] == "warn"
    assert warning_only["passed"] is True
    assert warning_only["summary"]["warnings"] == 1

    with_error = fv.run_checks(frame, [soft_expectation, hard_expectation])
    assert with_error["status"] == "fail"
    assert with_error["passed"] is False
    assert with_error["summary"]["warnings"] == 1
    assert with_error["summary"]["errors"] == 1
    assert {finding["code"] for finding in with_error["findings"]} == {
        "custom.soft_expectation",
        "custom.hard_expectation",
    }


def test_custom_check_exceptions_become_structured_failures():
    frame = pd.DataFrame({"value": [1, 2, 3]})

    @fv.check("exploding rule")
    def exploding_rule(df):
        raise RuntimeError("boom")

    result = fv.run_checks(frame, [exploding_rule])

    assert result["status"] == "fail"
    check_result = result["results"][0]
    assert check_result["passed"] is False
    assert check_result["severity"] == "error"
    assert "RuntimeError: boom" in check_result["execution_error"]
    assert "RuntimeError" in check_result["message"]


def test_each_custom_check_receives_an_isolated_dataframe_copy():
    frame = pd.DataFrame({"value": [1, 2, 3]})

    @fv.check("mutating check")
    def mutating_check(df):
        df["temporary"] = 1
        return True

    @fv.check("isolation check")
    def isolation_check(df):
        return "temporary" not in df.columns

    result = fv.run_checks(frame, [mutating_check, isolation_check])

    assert result["status"] == "pass"
    assert "temporary" not in frame.columns


def test_file_backed_custom_checks_disclose_full_materialization(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"value": [1, 2, 3]}).to_csv(path, index=False)

    result = fv.run_checks(path, [lambda df: bool((df["value"] > 0).all())])

    assert result["status"] == "pass"
    assert result["execution"]["full_materialization"] is True
    assert result["execution"]["source"]["kind"] == "file"


def test_gate_can_run_only_custom_warning_checks():
    frame = pd.DataFrame({"latency_ms": [10, 15, 20]})

    @fv.check("latency budget", severity="warning")
    def latency_budget(df):
        return {
            "passed": bool(df["latency_ms"].max() < 15),
            "message": "Latency exceeded the preferred budget.",
        }

    result = fv.gate(frame, custom_checks=[latency_budget])

    assert result["status"] == "warn"
    assert result["passed"] is True
    assert result["checks_run"] == ["custom"]
    assert result["checks"]["custom"]["status"] == "warn"
    assert result["execution"]["custom"]["method"] == "exact_custom_checks"
    assert result["execution"]["drift"] is None
    assert result["execution"]["validation"] is None
    assert "Latency exceeded the preferred budget." in result["reasons"]


def test_gate_custom_error_fails_quality_gate():
    frame = pd.DataFrame({"revenue": [100, -5, 80]})

    @fv.check("positive revenue")
    def positive_revenue(df):
        return {
            "passed": bool((df["revenue"] >= 0).all()),
            "message": "Negative revenue records are not allowed.",
        }

    result = fv.gate(frame, custom_checks=[positive_revenue])

    assert result["status"] == "fail"
    assert result["passed"] is False
    assert result["checks"]["custom"]["summary"]["errors"] == 1
    assert "Negative revenue records are not allowed." in result["reasons"]


def test_gate_still_requires_at_least_one_check_family():
    frame = pd.DataFrame({"value": [1]})

    with pytest.raises(ValueError, match="custom_checks"):
        fv.gate(frame)
