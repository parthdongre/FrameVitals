import subprocess
import sys


def test_profile_does_not_load_heavy_analysis_stack():
    code = r'''
import sys
import pandas as pd
import framevitals as fv

df = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "a"]})
result = fv.profile(df)
assert result["shape"]["rows"] == 3

for name in (
    "framevitals.pipeline",
    "framevitals.anomaly_ensemble",
    "framevitals.deep_statistics_v2",
    "sklearn",
    "statsmodels",
):
    assert name not in sys.modules, (name, sorted(k for k in sys.modules if k.startswith(name)))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
