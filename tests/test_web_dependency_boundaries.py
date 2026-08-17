import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("flask")


def test_web_app_does_not_eagerly_import_optional_ai_or_report_stack():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import app; "
                "assert 'framevitals.ai_agent' not in sys.modules; "
                "assert 'pydantic' not in sys.modules; "
                "assert 'framevitals.report_generator' not in sys.modules; "
                "assert 'framevitals.pdf_report_builder' not in sys.modules; "
                "assert 'framevitals.visualizer' not in sys.modules; "
                "assert 'framevitals.explainability' not in sys.modules; "
                "assert 'matplotlib' not in sys.modules; "
                "assert 'seaborn' not in sys.modules"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
