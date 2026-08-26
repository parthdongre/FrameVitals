from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pandas as pd
import pytest

import framevitals
from framevitals.config import AnalysisConfig
from framevitals.execution import ExecutionPolicy
from framevitals.execution_context import AnalysisContext, CONTEXT_SCHEMA_VERSION


def _context() -> AnalysisContext:
    return AnalysisContext(
        dataset_name="example",
        source={"kind": "dataframe", "rows": 3, "columns": 1},
        config=AnalysisConfig(mode="quick"),
        execution_policy=ExecutionPolicy(max_sample_rows=2),
        rows=3,
        columns=1,
    )


def test_context_cache_computes_once_and_tracks_hits():
    context = _context()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return {"value": 42}

    first = context.get_or_compute("profile", factory)
    second = context.get_or_compute("profile", factory)
    metadata = context.metadata()

    assert first is second
    assert calls == 1
    assert metadata["cache"]["entries"] == ["profile"]
    assert metadata["cache"]["misses"] == 1
    assert metadata["cache"]["hits"] == 1


def test_context_cache_prevents_duplicate_concurrent_work():
    context = _context()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        sleep(0.01)
        return object()

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(lambda _: context.get_or_compute("shared", factory), range(8)))

    assert calls == 1
    assert all(value is values[0] for value in values)
    assert context.metadata()["cache"]["hits"] == 7


def test_context_facts_and_samples_are_isolated_per_run():
    left = _context()
    right = _context()
    sample = pd.DataFrame({"x": [1, 2, 3]})

    left.set_fact("profile", {"shape": {"rows": 3, "columns": 1}})
    left.store_sample("working", sample, metadata={"purpose": "test"})

    assert left.require_fact("profile")["shape"]["rows"] == 3
    assert right.fact("profile") is None
    assert left.sample("working") is sample
    assert right.sample("working") is None

    metadata = left.metadata()
    assert metadata["facts"] == ["profile"]
    assert metadata["samples"]["working"] == {
        "purpose": "test",
        "rows": 3,
        "columns": 1,
    }
    assert "x" not in metadata["samples"]["working"]


def test_context_rejects_accidental_fact_or_sample_replacement():
    context = _context()
    context.set_fact("profile", 1)
    context.store_sample("working", [1, 2])

    with pytest.raises(KeyError, match="fact already exists"):
        context.set_fact("profile", 2)
    with pytest.raises(KeyError, match="sample already exists"):
        context.store_sample("working", [3])

    assert context.set_fact("profile", 2, overwrite=True) == 2
    assert context.store_sample("working", [3], overwrite=True) == [3]


def test_plan_surfaces_context_metadata_without_raw_sample_values():
    frame = pd.DataFrame({
        "x": list(range(20)),
        "y": [index * 2 for index in range(20)],
        "group": ["a", "b"] * 10,
    })

    plan = framevitals.plan(frame, mode="standard")
    metadata = plan["execution_context"]

    assert metadata["context_schema_version"] == CONTEXT_SCHEMA_VERSION
    assert metadata["shape"] == {"rows": 20, "columns": 3}
    assert metadata["samples"]["planning"]["rows"] == 20
    assert metadata["samples"]["planning"]["columns"] == 3
    assert set(metadata["facts"]) == {
        "column_roles",
        "execution_budget",
        "profile",
        "selection",
        "signals",
    }
    assert set(metadata["cache"]["entries"]) == {
        "column_roles",
        "dataset_signals",
        "execution_budget",
        "execution_plan",
    }
    assert metadata["cache"]["misses"] == 4
