import pandas as pd

import framevitals
from framevitals.quality_diagnostics import run_quality_diagnostics


def test_materialized_quality_diagnostics_respect_subten_hard_cap(monkeypatch):
    captured = {}

    def fake_quality(frame, **kwargs):
        captured["frame_rows"] = len(frame)
        captured["max_sample_rows"] = kwargs["max_sample_rows"]
        return {"available": True, "execution": {"sample_rows": kwargs["max_sample_rows"]}}

    monkeypatch.setattr(
        "framevitals.pipeline.run_quality_diagnostics",
        fake_quality,
    )

    frame = pd.DataFrame({
        "x": list(range(20)),
        "y": [index * 2 for index in range(20)],
    })
    framevitals.analyze(
        frame,
        mode="standard",
        max_sample_rows=4,
        disabled_modules=[
            "anomaly_detection",
            "time_series",
            "cleaning",
            "charts",
            "ai",
        ],
    )

    assert captured["frame_rows"] == 20
    assert captured["max_sample_rows"] == 4


def test_quality_diagnostics_execute_with_subten_cap():
    frame = pd.DataFrame({
        "x": list(range(20)),
        "y": [index * 2 for index in range(20)],
        "category": ["a", "b"] * 10,
    })

    result = run_quality_diagnostics(frame, max_sample_rows=4)

    assert result["available"] is True
    assert result["max_sample_rows"] == 4
