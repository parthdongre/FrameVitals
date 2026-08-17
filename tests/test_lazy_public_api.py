import subprocess
import sys


def test_top_level_import_keeps_heavy_implementation_lazy():
    script = r'''
import sys
import framevitals

required = (
    "profile",
    "roles",
    "health",
    "ml_readiness",
    "quality",
    "statistics",
    "anomalies",
    "target_analysis",
    "analyze",
    "compare",
    "validate",
    "gate",
)
for name in required:
    assert hasattr(framevitals, name), name

assert "framevitals.api" not in sys.modules
assert "framevitals.pipeline" not in sys.modules
assert "framevitals.cleaning_plan" not in sys.modules
assert "pandas" not in sys.modules
assert "numpy" not in sys.modules
'''

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
