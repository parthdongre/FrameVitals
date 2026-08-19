import subprocess
import sys

import pandas as pd

import framevitals as fv


def test_plan_exposes_execution_budget():
    frame = pd.DataFrame({
        "x": range(100),
        "y": range(100),
    })

    result = fv.plan(frame, mode="standard")

    assert result["execution_budget"]["rows"] == 100
    assert result["execution_budget"]["columns"] == 2
    assert result["selection"]["execution_budget"] == result["execution_budget"]
    assert result["source"]["format"] == "pandas"


def test_plan_classifies_extreme_shape_policy_without_changing_api():
    # The execution policy itself can reason about extreme shape without ever
    # allocating such a dataset. This is validated in adaptive-execution tests;
    # here we ensure ordinary plans expose the same policy schema.
    result = fv.plan(pd.DataFrame({"x": [1, 2, 3]}), mode="quick")

    budget = result["execution_budget"]
    assert budget["scale_class"] in {"normal", "large", "very_large", "extreme"}
    assert "max_memory_heavy_parallelism" in budget
    assert "bootstrap_sample_rows" in budget


def test_plan_does_not_import_full_execution_pipeline():
    code = r'''
import sys
import pandas as pd
import framevitals as fv

result = fv.plan(pd.DataFrame({"x": range(30), "y": range(30)}), mode="quick")
assert result["analysis_mode"] == "quick"
assert "framevitals.pipeline" not in sys.modules
assert "framevitals.anomaly_ensemble" not in sys.modules
assert "framevitals.deep_statistics_v2" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
