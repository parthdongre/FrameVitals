import numpy as np
import pandas as pd

from framevitals.advanced_indicators import (
    calculate_anomalies,
    calculate_freshness,
    detect_fairness_review,
)
from framevitals.sources import FileSource, PandasSource, resolve_source


def test_advanced_anomaly_scores_match_expected_iqr_density():
    # Keep IQR non-zero so the existing rule is applicable; the final row is an
    # obvious outlier in both columns and should therefore have density 1.0.
    frame = pd.DataFrame({
        "a": [0, 1, 2, 3, 100],
        "b": [1, 2, 3, 4, 50],
    })

    result = calculate_anomalies(frame)

    assert result["anomalous_rows"] == 1
    assert result["highest_score"] == 1.0
    assert result["top_rows"][0]["row_index"] == 4
    assert result["top_rows"][0]["score"] == 1.0


def test_sensitive_name_matching_uses_tokens_not_substrings():
    frame = pd.DataFrame({
        "average_score": [1, 2],
        "paid_amount": [5, 6],
        "customer_age": [20, 30],
    })

    result = detect_fairness_review(frame)

    assert result["needs_review"] is True
    assert result["columns"] == ["customer_age"]


def test_freshness_screens_candidates_before_full_parse():
    frame = pd.DataFrame({
        "identifier": [f"item-{i}" for i in range(500)],
        "event_date": pd.date_range("2025-01-01", periods=500, freq="D").astype(str),
    })

    result = calculate_freshness(frame)

    assert result["available"] is True
    assert result["date_column"] == "event_date"
    assert result["oldest_record"] == "2025-01-01"


def test_pandas_source_reports_materialized_metadata():
    frame = pd.DataFrame({"x": np.arange(10), "label": ["a"] * 10})
    source = PandasSource(frame)

    metadata = source.inspect()

    assert metadata.rows == 10
    assert metadata.columns == 2
    assert metadata.materialized is True
    assert metadata.supports_projection is True
    assert source.load().equals(frame)
    assert source.load() is not frame


def test_file_source_reports_cheap_metadata_without_loading(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    source = resolve_source(path)

    assert isinstance(source, FileSource)
    metadata = source.inspect()
    assert metadata.name == "data.csv"
    assert metadata.format == "csv"
    assert metadata.rows is None
    assert metadata.size_bytes == path.stat().st_size

    loaded = source.load()
    assert loaded.shape == (2, 2)


def test_resolve_source_rejects_unsupported_objects():
    try:
        resolve_source(object())
    except TypeError as exc:
        assert "DatasetSource" in str(exc)
    else:
        raise AssertionError("resolve_source should reject unsupported objects")
