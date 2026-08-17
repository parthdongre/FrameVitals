from framevitals.provenance import (
    EXECUTION_SCHEMA_VERSION,
    execution_provenance,
    load_fully_materializes,
    normalize_execution,
)
from framevitals.sources import DatasetMetadata


def test_execution_provenance_uses_shared_v1_contract():
    source = {
        "name": "data.parquet",
        "kind": "file",
        "format": "parquet",
        "rows": 10_000,
        "columns": 4,
        "size_bytes": 1234,
        "materialized": False,
        "supports_projection": True,
        "supports_streaming": True,
    }

    result = execution_provenance(
        "bounded_example",
        full_materialization=False,
        source=source,
        sampled=True,
        source_rows=10_000,
        source_columns=4,
        sample_rows=1_000,
        strategy="evenly_spaced",
        components={"missingness": "exact", "distribution": "sample"},
        reason="Bounded work.",
        scope="example",
        extra={"pair_budget": 20},
    )

    assert result["execution_schema_version"] == EXECUTION_SCHEMA_VERSION == "1"
    assert result["method"] == "bounded_example"
    assert result["full_materialization"] is False
    assert result["sampled"] is True
    assert result["source_rows"] == 10_000
    assert result["sample_rows"] == 1_000
    assert result["source"] == source
    assert result["components"]["missingness"] == "exact"
    assert result["pair_budget"] == 20


def test_execution_provenance_omits_unavailable_optional_values():
    result = execution_provenance(
        "schema_only",
        full_materialization=False,
    )

    assert result == {
        "execution_schema_version": "1",
        "method": "schema_only",
        "full_materialization": False,
    }


def test_normalize_execution_preserves_legacy_fields_and_adds_contract():
    result = normalize_execution({
        "scope": "bounded_deep_statistics",
        "sampled": True,
        "sample_rows": 500,
        "legacy_flag": "kept",
    })

    assert result["execution_schema_version"] == "1"
    assert result["method"] == "bounded_deep_statistics"
    assert result["scope"] == "bounded_deep_statistics"
    assert result["sampled"] is True
    assert result["sample_rows"] == 500
    assert result["legacy_flag"] == "kept"
    assert result["full_materialization"] is False


def test_materialization_semantics_are_not_file_specific():
    pandas_metadata = DatasetMetadata(
        name="<dataframe>",
        kind="memory",
        format="pandas",
        rows=3,
        columns=1,
        size_bytes=100,
        materialized=True,
        supports_projection=True,
        supports_streaming=False,
    )
    arrow_metadata = DatasetMetadata(
        name="<arrow_table>",
        kind="memory",
        format="arrow",
        rows=3,
        columns=1,
        size_bytes=24,
        materialized=True,
        supports_projection=True,
        supports_streaming=True,
    )
    relation_metadata = DatasetMetadata(
        name="<duckdb_relation>",
        kind="relation",
        format="duckdb",
        rows=3,
        columns=1,
        size_bytes=None,
        materialized=False,
        supports_projection=True,
        supports_streaming=True,
    )

    assert load_fully_materializes(pandas_metadata) is False
    assert load_fully_materializes(arrow_metadata) is True
    assert load_fully_materializes(relation_metadata) is True
