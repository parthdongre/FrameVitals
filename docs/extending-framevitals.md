# Extending FrameVitals

FrameVitals is designed to be extended at the edges rather than forked in the core.
There are two intentionally small extension boundaries:

1. **custom data checks** for domain invariants;
2. **`DatasetSource` implementations** for new storage/query systems.

Third-party check packages can additionally register checks through standard Python
entry points. Source adapters are normal Python objects implementing the source
protocol and can be shipped by any package without FrameVitals importing that package.

## Custom checks

Use `@framevitals.check` for application-owned invariants:

```python
import framevitals as fv

@fv.check(
    "positive revenue",
    severity="error",
    description="Revenue cannot be negative.",
)
def positive_revenue(df):
    minimum = float(df["revenue"].min())
    return {
        "passed": minimum >= 0,
        "message": "Negative revenue found." if minimum < 0 else "Revenue is valid.",
        "details": {"minimum": minimum},
    }
```

A check may return either a boolean-like value or a mapping containing `passed`.
Mappings can also include `message` and `details`.

Custom checks intentionally run against the complete DataFrame. FrameVitals cannot
safely infer whether an arbitrary Python invariant is sampleable, so it does not
silently weaken user-defined rules to bounded samples.

## Publishing a check plugin

A third-party package can expose checks through the `framevitals.checks` entry-point
group:

```toml
[project.entry-points."framevitals.checks"]
positive_revenue = "acme_framevitals_checks:positive_revenue"
```

The exported object can be a `DataCheck` or a compatible DataFrame callable.
Applications opt in explicitly:

```python
checks = fv.discover_checks()
result = fv.gate(current, custom_checks=checks)
```

FrameVitals never auto-loads installed check plugins. Loading an entry point executes
provider code, so discovery is an explicit trust decision.

### Plugin guidance

A good provider package should:

- keep public check names stable;
- avoid network or filesystem side effects during module import;
- return JSON-friendly `details` values;
- document whether a rule is a warning or a hard error;
- test against the supported FrameVitals version range;
- avoid mutating the DataFrame passed into a check.

FrameVitals gives every check an isolated DataFrame copy, but provider code should
still behave as a pure predicate where practical.

## Custom DatasetSource implementations

A storage/query integration does not need to be added to FrameVitals core. The minimal
protocol is deliberately small:

```python
from framevitals.sources import DatasetMetadata

class MySource:
    def inspect(self):
        return DatasetMetadata(
            name="warehouse.orders",
            kind="remote",
            format="my-engine",
            rows=1_000_000,
            columns=12,
            size_bytes=None,
            materialized=False,
            supports_projection=False,
            supports_streaming=False,
        )

    def load(self):
        # Return the complete dataset as pandas only when an operation requires it.
        return fetch_as_pandas()
```

Then pass the source directly:

```python
source = MySource()
print(fv.inspect_source(source))
report = fv.analyze(source, mode="quick")
```

Because the protocol is runtime-checkable, FrameVitals can consume compatible custom
objects without maintaining a registry of every provider package.

## Streaming source protocol

To opt into source-aware batch execution, additionally implement `iter_batches`:

```python
class MyStreamingSource(MySource):
    def iter_batches(self, *, batch_size=65_536, columns=None):
        ...
```

A streaming adapter should preserve these invariants:

1. `inspect()` reports the **true source shape**, not the retained sample shape.
2. `iter_batches()` respects `batch_size` and requested projection where advertised.
3. yielded batches are compatible with the Arrow-oriented streaming profiler.
4. `load()` remains available for exact/full-row APIs.
5. optional provider dependencies are imported lazily.
6. the adapter does not pretend an exact operation is streaming merely because the
   source supports batches.

## Prefer standard interoperability over one-off adapters

If a data library exposes the Arrow C Stream / PyCapsule interface, FrameVitals can
usually consume it through the generic Arrow boundary instead of carrying a dedicated
library dependency.

That is how FrameVitals can interoperate with modern dataframe producers such as
Polars while keeping Polars out of the runtime dependency graph.

Use a dedicated adapter when the storage/query system provides capabilities that the
generic Arrow boundary would lose. The DuckDB adapter is an example: it preserves a
lazy relation, pushes column projection into DuckDB, obtains exact relation metadata,
and streams Arrow batches without first constructing a complete in-memory table.

## Execution provenance for extensions

Extension authors should not invent incompatible exactness terminology. FrameVitals'
shared execution schema uses fields such as:

- `method`
- `full_materialization`
- `sampled`
- `source_rows`
- `source_columns`
- `sample_rows`
- `strategy`
- `components`
- `reason`

See [Execution provenance](execution-provenance.md) for the current schema.
