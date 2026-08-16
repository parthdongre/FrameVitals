import pytest

pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")

import framevitals as fv
from framevitals.sources import ArrowTableSource, resolve_source


def _frame(rows: int = 6_000):
    return pl.DataFrame({
        "value": [float(index) for index in range(rows)],
        "other": [float(index * 2) for index in range(rows)],
        "group": [f"g-{index % 5}" for index in range(rows)],
    })


def test_polars_dataframe_uses_generic_arrow_capsule_source():
    frame = _frame(2_000)

    source = resolve_source(frame)
    metadata = source.inspect()

    assert isinstance(source, ArrowTableSource)
    assert metadata.name == "<arrow_capsule_table>"
    assert metadata.kind == "memory"
    assert metadata.format == "arrow"
    assert metadata.rows == frame.height
    assert metadata.columns == frame.width
    assert metadata.supports_projection is True
    assert metadata.supports_streaming is True

    info = fv.inspect_source(frame)
    assert info == metadata.to_dict()


def test_polars_profile_stays_on_arrow_batch_path(monkeypatch):
    frame = _frame(12_000)

    def fail_load(self):
        raise AssertionError("Polars profiling must not materialize through pandas")

    monkeypatch.setattr(ArrowTableSource, "load", fail_load)
    result = fv.profile(frame)

    assert isinstance(result, fv.DiagnosticResult)
    assert result.diagnostic == "profile"
    assert result.dataset_name == "<arrow_capsule_table>"
    assert result["shape"] == {"rows": frame.height, "columns": frame.width}
    assert result["streaming_metadata"]["enabled"] is True
    assert result["streaming_metadata"]["full_materialization"] is False
    assert result["source_metadata"]["format"] == "arrow"


def test_polars_quick_analysis_stays_on_streaming_path(monkeypatch):
    frame = _frame(6_000)

    def fail_load(self):
        raise AssertionError("Polars analysis must not materialize through pandas")

    monkeypatch.setattr(ArrowTableSource, "load", fail_load)
    result = fv.analyze(frame, mode="quick", artifacts=False, workers=1)

    assert result["filename"] == "<arrow_capsule_table>"
    assert result["profile"]["shape"] == {
        "rows": frame.height,
        "columns": frame.width,
    }
    assert result["execution"]["streaming"]["enabled"] is True
    assert result["execution"]["streaming"]["full_materialization"] is False


def test_polars_bounded_statistics_do_not_load_full_dataframe(monkeypatch):
    frame = _frame(20_000)

    def fail_load(self):
        raise AssertionError("Polars statistics must remain on bounded Arrow batches")

    monkeypatch.setattr(ArrowTableSource, "load", fail_load)
    result = fv.statistics(frame, mode="quick", max_pairs=2)

    assert isinstance(result, fv.DiagnosticResult)
    assert result.execution["execution_schema_version"] == "1"
    assert result.execution["method"] == "bounded_deep_statistics"
    assert result.execution["full_materialization"] is False
    assert result.execution["source_rows"] == frame.height
    assert result.execution["sampled"] is True
