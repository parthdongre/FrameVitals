# FrameVitals

**FrameVitals is a source-aware data-health and quality-gate engine for tabular pipelines.**

It provides one workflow for inspecting data health, comparing production batches,
validating contracts, enforcing domain-specific invariants, and retaining compact
monitoring history.

```python
import framevitals as fv

source = fv.inspect_source("production.parquet")
report = fv.analyze("production.parquet", mode="quick")
drift = fv.compare("training.parquet", "production.parquet")
validation = fv.validate("production.parquet", contract)
gate = fv.gate(
    "production.parquet",
    reference="training.parquet",
    contract=contract,
)
```

## Core design

FrameVitals is built around four constraints:

1. **Source-aware execution** — supported sources expose metadata, projection, and
   batches before FrameVitals decides whether pandas materialization is necessary.
2. **Bounded expensive work** — statistics, anomaly detection, relationship discovery,
   and drift use explicit execution budgets when full-row computation is not required.
3. **Exactness where correctness requires it** — contracts and arbitrary custom Python
   invariants are not silently weakened to sampled checks.
4. **Execution transparency** — public results disclose whether work was exact,
   sampled, estimated, streamed, or fully materialized.

## Main workflow

```text
source
  │
  ├─► inspect_source
  │
  ├─► analyze ─► snapshot ─► SnapshotHistory
  │
  ├─► compare(reference, current)
  │
  ├─► validate(current, contract)
  │
  └─► gate
        ├─ contract validation
        ├─ drift
        └─ custom checks / plugins
```

## Supported source families

FrameVitals currently recognizes:

- pandas `DataFrame` inputs;
- CSV and TSV files, with optional Arrow streaming;
- Parquet through the Arrow capability;
- PyArrow `Table` and `RecordBatch` inputs;
- table producers supporting the Arrow C Stream / PyCapsule interface;
- lazy DuckDB relations through the optional DuckDB adapter;
- custom objects implementing the FrameVitals `DatasetSource` protocol.

Use `fv.inspect_source(data)` before analysis when you want to see the source's
shape metadata and streaming/projection capabilities.

## Installation

```bash
pip install framevitals
```

Optional capabilities are deliberately separated:

```bash
pip install "framevitals[arrow]"
pip install "framevitals[duckdb]"
pip install "framevitals[excel]"
pip install "framevitals[plot]"
pip install "framevitals[ml]"
pip install "framevitals[ai]"
pip install "framevitals[web]"
```

For documentation development:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Documentation map

- [Source-aware execution](source-execution.md)
- [Execution provenance](execution-provenance.md)
- [Quality gates and custom checks](quality-gates.md)
