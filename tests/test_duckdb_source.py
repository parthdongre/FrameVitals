import pytest

pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

import framevitals
from framevitals.duckdb_source import DuckDBRelationSource
from framevitals.sources import resolve_source


def _relation(rows: int = 12_000):
    connection = duckdb.connect()
    relation = connection.sql(
        f"""
        SELECT
            range::DOUBLE AS value,
            (range * 2)::DOUBLE AS other,
            concat('g-', range % 5) AS grp
        FROM range({int(rows)})
        """
    )
    return connection, relation


def test_duckdb_relation_source_exposes_exact_metadata_projection_and_batches():
    connection, relation = _relation(4_000)
    try:
        source = resolve_source(relation)
        assert isinstance(source, DuckDBRelationSource)

        metadata = source.inspect()
        assert metadata.name == "<duckdb_relation>"
        assert metadata.kind == "relation"
        assert metadata.format == "duckdb"
        assert metadata.rows == 4_000
        assert metadata.columns == 3
        assert metadata.size_bytes is None
        assert metadata.materialized is False
        assert metadata.supports_projection is True
        assert metadata.supports_streaming is True
        assert source.schema().names == ["value", "other", "grp"]

        batches = list(source.iter_batches(batch_size=700, columns=["value", "grp"]))
        assert sum(batch.num_rows for batch in batches) == 4_000
        assert max(batch.num_rows for batch in batches) <= 700
        assert batches[0].schema.names == ["value", "grp"]
    finally:
        connection.close()


def test_public_profile_streams_duckdb_relation_without_pandas_materialization(monkeypatch):
    connection, relation = _relation(12_000)

    def fail_load(self):
        raise AssertionError("DuckDB profile must not materialize the complete relation")

    monkeypatch.setattr(DuckDBRelationSource, "load", fail_load)
    try:
        result = framevitals.profile(relation)
    finally:
        connection.close()

    assert result["dataset_name"] == "<duckdb_relation>"
    assert result["shape"] == {"rows": 12_000, "columns": 3}
    assert result["streaming_metadata"]["enabled"] is True
    assert result["streaming_metadata"]["full_materialization"] is False
    assert result["source_metadata"]["kind"] == "relation"
    assert result["source_metadata"]["format"] == "duckdb"


def test_public_analyze_streams_duckdb_relation(monkeypatch):
    connection, relation = _relation(6_000)

    def fail_load(self):
        raise AssertionError("DuckDB analysis must stay on the streaming relation path")

    monkeypatch.setattr(DuckDBRelationSource, "load", fail_load)
    try:
        result = framevitals.analyze(
            relation,
            mode="quick",
            artifacts=False,
            workers=1,
        )
    finally:
        connection.close()

    assert result["filename"] == "<duckdb_relation>"
    assert result["profile"]["shape"] == {"rows": 6_000, "columns": 3}
    assert result["execution"]["streaming"]["enabled"] is True
    assert result["execution"]["streaming"]["full_materialization"] is False
    assert result["execution"]["streaming"]["source_rows"] == 6_000


def test_plan_uses_bounded_duckdb_relation_sample(monkeypatch):
    connection, relation = _relation(20_000)

    def fail_load(self):
        raise AssertionError("DuckDB planning must not load the complete relation")

    monkeypatch.setattr(DuckDBRelationSource, "load", fail_load)
    try:
        plan = framevitals.plan(relation, mode="standard")
    finally:
        connection.close()

    assert plan["dataset_name"] == "<duckdb_relation>"
    assert plan["source"]["format"] == "duckdb"
    assert plan["shape"] == {"rows": 20_000, "columns": 3}
    assert plan["planning_data"]["materialized_full_dataset"] is False
    assert plan["planning_data"]["sampled"] is True
    assert plan["planning_data"]["sample_rows"] == 5_000
