import os
import subprocess
import sys
from pathlib import Path


def test_compat_api_import_does_not_eagerly_load_data_stack():
    src_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import framevitals.api; "
                "assert 'pandas' not in sys.modules; "
                "assert 'framevitals.pipeline' not in sys.modules; "
                "assert 'framevitals.anomaly_ensemble' not in sys.modules"
            ),
        ],
        cwd=src_root.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_compat_api_delegates_compare_to_operations(monkeypatch):
    import framevitals.api as api
    import framevitals.operations as operations

    expected = {"available": True, "marker": "canonical-compare"}

    def fake_compare(reference, current, *, columns=None, max_columns=30):
        assert reference == "reference"
        assert current == "current"
        assert columns == ["value"]
        assert max_columns == 4
        return expected

    monkeypatch.setattr(operations, "compare", fake_compare)
    result = api.compare(
        "reference",
        "current",
        columns=["value"],
        max_columns=4,
    )

    assert result is expected


def test_compat_api_delegates_analyze_to_source_dispatcher(monkeypatch):
    import framevitals.analysis_api as analysis_api
    import framevitals.api as api

    expected = {"marker": "canonical-analysis"}

    def fake_analyze(data, **kwargs):
        assert data == "dataset.csv"
        assert kwargs["mode"] == "quick"
        assert kwargs["artifacts"] is False
        return expected

    monkeypatch.setattr(analysis_api, "analyze", fake_analyze)
    result = api.analyze("dataset.csv", mode="quick", artifacts=False)

    assert result is expected
