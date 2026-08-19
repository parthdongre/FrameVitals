# Source-aware execution

FrameVitals separates **what the data is stored in** from **what a diagnostic needs to
compute**. Public APIs resolve inputs into a `DatasetSource` before choosing between
batch streaming and pandas materialization.

## Inspect a source first

```python
import framevitals as fv

info = fv.inspect_source(data)
print(info)
```

Typical fields include:

```json
{
  "name": "production.parquet",
  "kind": "file",
  "format": "parquet",
  "rows": 1200000,
  "columns": 42,
  "size_bytes": 84599210,
  "materialized": false,
  "supports_projection": true,
  "supports_streaming": true
}
```

`inspect_source()` does not run health/statistical diagnostics. Some adapters may need
to scan lightweight metadata (for example exact relation row counts), but they do not
materialize the full dataset into pandas simply to describe source capabilities.

## DatasetSource protocol

A minimal source provides:

```python
class DatasetSource(Protocol):
    def inspect(self) -> DatasetMetadata: ...
    def load(self) -> pandas.DataFrame: ...
```

A streaming source additionally provides:

```python
class StreamingDatasetSource(DatasetSource, Protocol):
    def iter_batches(
        self,
        *,
        batch_size: int = 65_536,
        columns: Sequence[str] | None = None,
    ) -> Iterator[Any]: ...
```

This means new storage systems can integrate with FrameVitals without changing every
diagnostic implementation.

## Current adapters

### pandas

A pandas `DataFrame` is already materialized. It remains the reference in-memory
representation for exact/full-row operations.

### Parquet

With `framevitals[arrow]`, Parquet supports:

- exact row/column metadata from file metadata;
- column projection;
- Arrow record batches;
- bounded global row sampling without `ParquetSource.load()`.

### CSV / TSV

With Arrow installed, compatible delimited files use incremental Arrow CSV reading.
If Arrow cannot interpret a file, FrameVitals preserves compatibility by falling back
to the existing pandas loader rather than silently returning a different parse.

### PyArrow Table and RecordBatch

PyArrow in-memory objects enter the batch path directly. FrameVitals does not first
convert the complete object to pandas for streaming-safe diagnostics.

### Arrow C Stream / PyCapsule producers

Table-like objects implementing `__arrow_c_stream__` can be normalized through
PyArrow. This gives interoperable support to producers such as modern dataframe
libraries without hard-coding each library into FrameVitals.

A raw `RecordBatchReader` is currently rejected because it does not expose the cheap
exact row count required by FrameVitals' current planning/profiling contract.

### DuckDB relations

With `framevitals[duckdb]`, a lazy `DuckDBPyRelation` supports:

- exact row count via a cached aggregate;
- schema inspection;
- projection pushed into the DuckDB relation;
- Arrow `RecordBatchReader` transport;
- pandas materialization only when an exact/full-row API explicitly requires it.

```python
import duckdb
import framevitals as fv

con = duckdb.connect()
relation = con.sql("SELECT * FROM read_parquet('events/*.parquet')")

print(fv.inspect_source(relation))
report = fv.analyze(relation, mode="quick")
```

## Source-aware public APIs

The source abstraction is shared by:

- `inspect_source()`
- `profile()`
- `roles()`
- `health()`
- `ml_readiness()`
- `quality()`
- `statistics()`
- `anomalies()`
- `relationships()`
- `target_analysis()`
- `analyze()`
- `plan()`
- `compare()`
- `validate()`
- `run_checks()`
- `gate()`

The operation still decides whether streaming is semantically valid. For example,
contract validation and arbitrary Python checks deliberately remain exact.

## Adding a source adapter

Contributors should preserve these invariants:

1. `inspect()` should be cheaper/safer than full pandas materialization.
2. `iter_batches()` must respect requested projection where supported.
3. source metadata must report the **true** source shape, not sample shape.
4. bounded algorithms must disclose sampling strategy and row counts.
5. exact operations must not claim streaming merely because the source supports it.
6. optional storage dependencies must remain lazily imported.

See [Execution provenance](execution-provenance.md) for the metadata contract used to
report these choices.
