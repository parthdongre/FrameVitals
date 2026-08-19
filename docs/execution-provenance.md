# Execution provenance

FrameVitals results can mix exact metadata, full-stream calculations, bounded row
samples, estimates, and operations that intentionally materialize a complete pandas
DataFrame. The `execution` block makes those decisions machine-readable.

## Schema version

The shared execution contract currently uses:

```json
{
  "execution_schema_version": "1"
}
```

During the `0.x` series, older operation-specific keys remain available while public
results converge on this common vocabulary.

## Common fields

| Field | Meaning |
| --- | --- |
| `execution_schema_version` | Version of the shared execution metadata contract |
| `method` | Stable description of the execution strategy |
| `full_materialization` | Whether FrameVitals created a complete pandas representation of a non-pandas source |
| `source` | Source metadata/capabilities when available |
| `sampled` | Whether row-level work used fewer than all source rows |
| `source_rows` | Exact source row count when known |
| `source_columns` | Exact source column count when known |
| `sample_rows` | Number of rows used for bounded row-level work |
| `strategy` | Sampling or source-consumption strategy |
| `components` | Per-component exactness/approximation information |
| `reason` | Human-readable explanation for the execution choice |
| `scope` | Legacy/operation-specific scope retained during the `0.x` migration |

Not every field appears in every result. Missing information is omitted rather than
represented as ambiguous placeholder values.

## Full materialization semantics

`full_materialization` describes **what the operation did**, not merely where the
source came from.

A pandas `DataFrame` is already materialized when supplied by the caller, so running
an exact operation on it does not count as FrameVitals materializing a new source.
By contrast, converting any of these into a complete pandas DataFrame does count:

- a file-backed source;
- a PyArrow table;
- a DuckDB relation;
- a remote/custom source.

This distinction matters for memory planning and for pipelines that deliberately stay
on Arrow/relation-backed execution paths.

## Examples

### Bounded statistics

```json
{
  "execution_schema_version": "1",
  "method": "bounded_deep_statistics",
  "full_materialization": false,
  "sampled": true,
  "source_rows": 120000,
  "source_columns": 18,
  "sample_rows": 1000,
  "strategy": "streaming_evenly_spaced_global_rows"
}
```

### Exact contract validation

```json
{
  "execution_schema_version": "1",
  "method": "exact_contract_validation",
  "full_materialization": true,
  "sampled": false,
  "source": {
    "kind": "relation",
    "format": "duckdb"
  }
}
```

### Quality gate

A gate aggregates the execution blocks of the selected check families:

```python
result = fv.gate(
    current,
    reference=reference,
    contract=contract,
    custom_checks=[positive_revenue],
)

print(result["execution"])
```

The top-level gate reports `full_materialization=True` when **any** selected check
family required it, while nested `validation`, `drift`, and `custom` entries preserve
family-level provenance.

## Exactness is diagnostic-specific

A streaming-capable source does not make every metric approximate. For example:

- source shape and schema can be exact;
- missingness can be computed over the full stream;
- duplicate rate may become a bounded estimate on a large source;
- deep statistics may use a deterministic bounded sample;
- contract uniqueness checks stay exact and therefore may materialize.

Consumers should prefer the `execution`/`components` fields over assumptions based on
file format or dataset size.
