import subprocess
import sys


def test_contract_and_drift_operations_do_not_load_full_pipeline():
    code = r'''
import sys
import pandas as pd
import framevitals as fv

reference = pd.DataFrame({"x": range(30), "label": ["a", "b"] * 15})
current = reference.copy()

contract = fv.infer_contract(reference)
validation = fv.validate(current, contract)
drift = fv.compare(reference, current)
gate = fv.gate(current, reference=reference, contract=contract)

assert validation["status"] in {"pass", "warn"}
assert drift["available"] is True
assert gate["passed"] is True

for name in (
    "framevitals.pipeline",
    "framevitals.anomaly_ensemble",
    "framevitals.deep_statistics_v2",
    "framevitals.model_leaderboard",
):
    assert name not in sys.modules, name
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
