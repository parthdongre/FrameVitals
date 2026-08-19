import numpy as np
import pandas as pd
import pytest

import framevitals as fv
from framevitals.relationship_graph import build_numeric_relationship_graph


def test_relationship_graph_finds_strong_cluster_without_dense_matrix():
    rng = np.random.default_rng(7)
    rows = 600
    base = rng.normal(size=rows)
    frame = {
        "base": base,
        "base_copy": base.copy(),
        "base_negative": -base,
    }
    for index in range(57):
        frame[f"noise_{index}"] = rng.normal(size=rows)
    dataframe = pd.DataFrame(frame)

    result = build_numeric_relationship_graph(
        dataframe,
        max_sample_rows=256,
        min_abs_correlation=0.95,
    )

    assert result["available"] is True
    assert result["nodes"] == 60
    assert result["method"] == "bounded_simhash_lsh_then_pearson"

    pairs = {
        frozenset((edge["source"], edge["target"]))
        for edge in result["edges"]
    }
    assert frozenset(("base", "base_copy")) in pairs
    assert frozenset(("base", "base_negative")) in pairs

    candidate = result["candidate_generation"]
    assert candidate["candidate_pairs"] < candidate["total_possible_dense_pairs"]
    assert candidate["dense_pairs_avoided"] > 0

    largest = result["graph"]["components"][0]
    assert largest["size"] >= 3
    assert {"base", "base_copy", "base_negative"}.issubset(largest["members"])


def test_relationship_graph_is_row_bounded():
    rng = np.random.default_rng(11)
    dataframe = pd.DataFrame(
        rng.normal(size=(5_000, 40)),
        columns=[f"x_{index}" for index in range(40)],
    )

    result = build_numeric_relationship_graph(dataframe, max_sample_rows=128)

    assert result["sample"]["source_rows"] == 5_000
    assert result["sample"]["sample_rows"] == 128
    assert result["sample"]["sampled"] is True


def test_relationship_graph_candidate_budget_is_explicit():
    rng = np.random.default_rng(3)
    base = rng.normal(size=200)
    dataframe = pd.DataFrame({
        f"copy_{index}": base + rng.normal(scale=1e-8, size=200)
        for index in range(100)
    })

    result = build_numeric_relationship_graph(
        dataframe,
        max_sample_rows=128,
        max_candidate_pairs=50,
        min_abs_correlation=0.90,
    )

    assert result["candidate_generation"]["candidate_pairs"] <= 50
    assert result["candidate_generation"]["truncated"] is True
    assert result["verification"]["verified_relationships"] > 0


def test_relationship_graph_rejects_invalid_controls():
    dataframe = pd.DataFrame({"a": range(30), "b": range(30)})

    with pytest.raises(ValueError, match="max_sample_rows"):
        build_numeric_relationship_graph(dataframe, max_sample_rows=10)
    with pytest.raises(ValueError, match="projections"):
        build_numeric_relationship_graph(dataframe, projections=17)
    with pytest.raises(ValueError, match="min_abs_correlation"):
        build_numeric_relationship_graph(dataframe, min_abs_correlation=0)


def test_relationship_graph_handles_insufficient_numeric_columns():
    result = build_numeric_relationship_graph(
        pd.DataFrame({"label": ["a", "b", "c"]})
    )

    assert result["available"] is False
    assert result["nodes"] == 0
    assert result["edges"] == []


def test_public_relationships_api_preserves_dataset_name(tmp_path):
    path = tmp_path / "related.csv"
    dataframe = pd.DataFrame({
        "a": np.arange(100, dtype=float),
        "b": np.arange(100, dtype=float) * 2,
        "noise": np.random.default_rng(9).normal(size=100),
    })
    dataframe.to_csv(path, index=False)

    result = fv.relationships(path, min_abs_correlation=0.95)

    assert result["dataset_name"] == "related.csv"
    assert result["available"] is True
    pairs = {
        frozenset((edge["source"], edge["target"]))
        for edge in result["edges"]
    }
    assert frozenset(("a", "b")) in pairs
