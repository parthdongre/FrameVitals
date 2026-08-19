from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import framevitals as fv
from framevitals.fast_anomaly import fast_anomaly_scan
from framevitals.relationship_graph import build_numeric_relationship_graph
from framevitals.semantic_types import infer_semantic_types
from framevitals.target_intelligence import run_target_intelligence


def _mixed_frame(rows: int = 12_000) -> pd.DataFrame:
    rng = np.random.default_rng(20260817)
    linear = np.arange(rows, dtype=np.float64)
    correlated = linear * 3.0 + 7.0
    noisy_signal = linear * 0.25 + rng.normal(scale=5.0, size=rows)
    normal = rng.normal(loc=4.0, scale=1.5, size=rows)
    exponential = rng.exponential(scale=2.0, size=rows)

    missing_numeric = normal.copy()
    missing_numeric[::101] = np.nan

    category = np.array(["alpha", "beta", "gamma", "delta"] * (rows // 4), dtype=object)
    category[::97] = None

    target = (linear % 10 >= 2).astype(np.int8)

    return pd.DataFrame({
        "linear": linear,
        "correlated": correlated,
        "noisy_signal": noisy_signal,
        "normal": normal,
        "exponential": exponential,
        "missing_numeric": missing_numeric,
        "category": category,
        "event_time": pd.date_range("2026-01-01", periods=rows, freq="min"),
        "target": target,
    })


def _write_mixed_parquet(path, rows: int = 12_000) -> pd.DataFrame:
    frame = _mixed_frame(rows)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=777)
    return frame


def test_mixed_parquet_profile_matches_ground_truth(tmp_path):
    path = tmp_path / "mixed-ground-truth.parquet"
    frame = _write_mixed_parquet(path)

    result = fv.profile(path)

    assert result["shape"] == {"rows": len(frame), "columns": len(frame.columns)}
    assert result["streaming_metadata"]["enabled"] is True
    assert result["streaming_metadata"]["full_materialization"] is False
    assert result["missing_counts"]["missing_numeric"] == int(frame["missing_numeric"].isna().sum())
    assert result["missing_counts"]["category"] == int(frame["category"].isna().sum())
    assert result["numeric_summary"]["linear"]["mean"] == pytest.approx(
        round(float(frame["linear"].mean()), 3)
    )
    assert result["numeric_summary"]["correlated"]["min"] == pytest.approx(7.0)
    assert result["numeric_summary"]["correlated"]["max"] == pytest.approx(
        float(frame["correlated"].max())
    )
    assert result["numeric_summary"]["linear"]["50%"] == pytest.approx(
        float(frame["linear"].median()), rel=0.03
    )
    assert "category" in result["categorical_summary"]
    assert "event_time" in result["date_columns"]


def test_mixed_parquet_full_public_analysis_keeps_streaming_and_target_intelligence(tmp_path):
    path = tmp_path / "mixed-analysis.parquet"
    frame = _write_mixed_parquet(path)

    result = fv.analyze(
        path,
        target="target",
        mode="quick",
        artifacts=False,
        workers=2,
    )

    assert result["profile"]["shape"] == {"rows": len(frame), "columns": len(frame.columns)}
    streaming = result["execution"]["streaming"]
    assert streaming["enabled"] is True
    assert streaming["full_materialization"] is False
    assert streaming["single_full_source_profile_scan"] is True
    assert result["target_intelligence"]["available"] is True
    assert result["target_intelligence"]["target_column"] == "target"
    assert result["target_intelligence"]["task_type"] == "classification"
    assert 0 <= result["health"]["overall_score"] <= 100
    assert 0 <= result["ml_readiness"]["score"] <= 100


def test_known_numeric_relationship_is_found_without_dense_matrix():
    frame = _mixed_frame(2_000)[["linear", "correlated", "noisy_signal", "normal"]]
    result = build_numeric_relationship_graph(
        frame,
        max_sample_rows=512,
        min_abs_correlation=0.98,
    )

    pairs = {
        frozenset((edge["source"], edge["target"]))
        for edge in result["edges"]
    }
    assert frozenset(("linear", "correlated")) in pairs
    assert result["candidate_generation"]["candidate_pairs"] < result["candidate_generation"][
        "total_possible_dense_pairs"
    ]


def test_injected_multivariate_anomalies_have_high_recall():
    rng = np.random.default_rng(77)
    rows = 3_000
    matrix = rng.normal(size=(rows, 10))
    injected = set(range(rows - 16, rows))
    matrix[-16:] += 10.0
    frame = pd.DataFrame(matrix, columns=[f"x{index}" for index in range(matrix.shape[1])])

    result = fast_anomaly_scan(frame, contamination=0.02, top_k=30)
    top_indices = {int(item["row_index"]) for item in result["top_rows"][:25]}

    assert result["available"] is True
    assert len(top_indices & injected) >= 12


def test_target_intelligence_finds_numeric_and_categorical_leakage():
    target = [0, 1] * 200
    frame = pd.DataFrame({
        "numeric_signal": [value * 10 + (index % 3) for index, value in enumerate(target)],
        "segment": ["stay" if value == 0 else "leave" for value in target],
        "noise": np.random.default_rng(9).normal(size=len(target)),
        "target": target,
    })

    result = run_target_intelligence(frame, target_column="target")
    associations = {item["feature"]: item for item in result["top_associations"]}
    leakage = {item["feature"] for item in result["leakage"]["warnings"]}

    assert result["task_type"] == "classification"
    assert associations["numeric_signal"]["score"] > 0.9
    assert associations["segment"]["score"] > 0.9
    assert "segment" in leakage


def test_semantic_type_ground_truth_for_sensitive_string_patterns():
    cases = {
        "email": pd.Series([f"user{index}@example.com" for index in range(40)]),
        "url": pd.Series([f"https://example.com/{index}" for index in range(40)]),
        "ip_address": pd.Series([f"10.0.0.{index + 1}" for index in range(40)]),
    }

    for expected, series in cases.items():
        result = infer_semantic_types(series, max_samples=40)
        assert result["primary"] == expected
        assert result["candidates"][0]["confidence"] >= 0.7
