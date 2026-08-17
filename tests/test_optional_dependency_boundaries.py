import os
import subprocess
import sys
from pathlib import Path

import pytest

from framevitals.loader import load_dataset


def _run_import_assertion(code: str):
    src_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=src_root.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_core_import_does_not_eagerly_load_ai_dependency():
    completed = _run_import_assertion(
        "import sys; import framevitals; assert 'pydantic' not in sys.modules"
    )
    assert completed.returncode == 0, completed.stderr


def test_pipeline_import_does_not_eagerly_load_plotting_stack():
    completed = _run_import_assertion(
        (
            "import sys; import framevitals.pipeline; "
            "assert 'framevitals.visualizer' not in sys.modules; "
            "assert 'framevitals.explainability' not in sys.modules; "
            "assert 'matplotlib' not in sys.modules; "
            "assert 'seaborn' not in sys.modules"
        )
    )
    assert completed.returncode == 0, completed.stderr


def test_explainability_import_does_not_require_plotting_stack():
    completed = _run_import_assertion(
        (
            "import sys; import framevitals.explainability; "
            "assert 'matplotlib' not in sys.modules; "
            "assert 'seaborn' not in sys.modules"
        )
    )
    assert completed.returncode == 0, completed.stderr


def test_excel_loader_reports_framevitals_extra(monkeypatch, tmp_path):
    path = tmp_path / "dataset.xlsx"
    path.write_bytes(b"placeholder")

    def missing_engine(*args, **kwargs):
        raise ImportError("missing Excel engine")

    monkeypatch.setattr("framevitals.loader.pd.read_excel", missing_engine)

    with pytest.raises(ImportError, match=r"framevitals\[excel\]"):
        load_dataset(path)
