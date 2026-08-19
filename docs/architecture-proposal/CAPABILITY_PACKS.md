# FrameVitals Capability Pack Specification

This document defines how optional capabilities can grow without turning the base install into a huge dependency bundle.

## Design goals

- The base `framevitals` install must remain useful by itself.
- Optional packs should add clearly defined capabilities.
- Installation and enablement are separate states.
- Model weights should not be bundled into PyPI wheels unless tiny and justified.
- Deep-learning frameworks must stay optional.
- Every optional capability must fail gracefully and preserve deterministic fallbacks when possible.

## Proposed packs

### `core`
Installed by default.

Scope:
- CSV/TSV/basic tabular loading
- structural profiling
- data quality
- health score
- core statistics
- contracts/validation
- drift basics
- result schema
- standard CLI

Goal: a strong data-analysis/data-health library even with no extras.

---

### `viz`
Purpose: rendering and rich human-facing reports.

Potential contents:
- plotting dependencies
- standalone HTML report helpers
- PDF/chart rendering
- notebook visual components

CLI/TUI label: **Visual Reports**

---

### `excel`
Purpose: XLS/XLSX ingestion.

Potential contents:
- `openpyxl`
- `xlrd` where still required

CLI/TUI label: **Excel Support**

---

### `ml`
Purpose: classical ML-based diagnostics.

Potential capabilities:
- baseline models
- feature importance
- model diagnostics
- Isolation Forest
- LOF
- ECOD/COPOD/HBOS or equivalent anomaly methods
- optional gradient boosting diagnostics
- optional SHAP-style explanation layer

This pack is diagnostic, not a general AutoML system.

CLI/TUI label: **ML Diagnostics**

---

### `deep`
Purpose: optional nonlinear/deep diagnostics where they add real value.

Potential capabilities:
- MLP autoencoder anomaly detector
- DeepSVDD-style anomaly detector
- variational autoencoder experiment path
- small temporal CNN/TCN models
- optional tabular nonlinear baseline

Framework policy:
- do not require a deep-learning framework in the core package
- choose one supported runtime initially rather than supporting everything
- CPU must remain supported for small models where practical
- GPU use should be opt-in/auto-detected and never required for ordinary FrameVitals analysis

CLI/TUI label: **Deep Models**

---

### `text`
Purpose: advanced text and semantic diagnostics.

Potential capabilities:
- embedding-based text drift
- semantic similarity/grouping
- advanced PII helpers
- language detection
- text quality/duplication signals

Large embedding weights should be managed by the model registry, not bundled directly into the wheel.

CLI/TUI label: **Advanced Text / NLP**

---

### `polars`
Purpose: Polars input/backend support.

Potential capabilities:
- DataFrame/LazyFrame adapters
- projection/predicate-aware analysis paths where practical
- backend-native operations for large data

CLI/TUI label: **Polars Backend**

---

### `arrow`
Purpose: PyArrow and columnar I/O.

Potential capabilities:
- Arrow Table/RecordBatch support
- Parquet scans
- batch/streaming analysis paths
- projection/filter support

CLI/TUI label: **Arrow / Parquet Backend**

---

### `sql`
Purpose: database/query-backed analysis.

Potential capabilities:
- DuckDB adapter
- SQLAlchemy-compatible sources where useful
- pushdown of lightweight aggregates/projections

The initial goal should be analysis of query results and large local files rather than becoming a database management tool.

CLI/TUI label: **SQL Adapters**

---

### `cloud`
Purpose: remote object/file storage integration.

Potential capabilities:
- fsspec-based inputs
- S3-compatible paths
- selected cloud filesystem adapters

Credentials must be supplied by the user's environment/provider tooling; FrameVitals should not invent its own secret store.

CLI/TUI label: **Cloud Filesystems**

---

### `tui`
Purpose: interactive terminal application if kept separate from core.

Potential contents:
- terminal UI framework
- interactive tables/forms/progress views

If the dependency footprint is small enough, this pack may eventually become part of the default install. Until then, the standard argparse CLI remains available without it.

CLI/TUI label: **Interactive Terminal UI**

---

### `ai`
Purpose: optional natural-language interpretation/agent features.

Rules:
- deterministic analysis remains the source of factual metrics
- AI summarizes/interprets structured findings
- no AI dependency required for normal analysis
- provider-specific behavior stays isolated

CLI/TUI label: **AI Interpretation**

## Example extras layout

Conceptually:

```toml
[project.optional-dependencies]
viz = ["..."]
excel = ["..."]
ml = ["..."]
deep = ["..."]
text = ["..."]
polars = ["..."]
arrow = ["..."]
sql = ["..."]
cloud = ["..."]
tui = ["..."]
ai = ["..."]
```

The exact dependency choices should be benchmarked and reviewed before modifying `pyproject.toml`.

## Capability registry

FrameVitals should expose capability metadata independently from the Python packaging implementation.

Example conceptual record:

```python
Capability(
    id="deep",
    name="Deep Models",
    installed=False,
    enabled=False,
    extra="deep",
    provides=[
        "anomaly.autoencoder",
        "anomaly.deep_svdd",
        "timeseries.tcn",
    ],
    resource_class="high",
)
```

This registry powers:
- `framevitals addons list`
- TUI install/toggle screens
- planner applicability checks
- `framevitals doctor`
- error messages when an optional analysis was explicitly requested but is unavailable

## Installed vs enabled

These states must not be conflated.

| Installed | Enabled | Meaning |
|---|---|---|
| No | No | capability unavailable |
| Yes | No | dependency exists but planner will not use it automatically |
| Yes | Yes | planner may use capability when applicable |
| No | Yes | invalid state; config should warn and treat as unavailable |

Users should be able to keep a pack installed but disabled for speed/reproducibility.

## Model registry

Model files should have a separate lifecycle from Python packages.

Suggested cache organization:

```text
<framevitals-data-dir>/
  models/
    semantic-types-small/
      1.0.0/
        model...
        manifest.json
    temporal-anomaly-tiny/
      1.0.0/
        model...
        manifest.json
```

Each manifest should include:
- model ID/version
- FrameVitals compatibility
- task
- framework/runtime
- checksum
- source URL
- license
- file sizes
- expected input contract

## Download policy

Model download should occur only through explicit actions such as:

```bash
framevitals models install semantic-types-small
```

or the equivalent TUI **Download** button.

Ordinary analysis can suggest a model but should not automatically download it.

Example:

```text
Semantic type confidence is low for 3 columns.
Optional model 'semantic-types-small' could provide a second opinion.
[ Download & enable ] [ Ignore ]
```

Even there, user confirmation is required.

## Reproducibility

Every analysis result should record optional capability/model metadata when used:
- pack ID/version
- relevant dependency versions
- model ID/version/checksum
- backend
- random seed
- device (CPU/GPU)

This makes deep/model-assisted diagnostics reproducible and explainable.

## Dependency policy

Before adding a package to any optional pack, evaluate:
1. user value
2. wheel size and install reliability
3. Python/platform compatibility
4. maintenance activity
5. license compatibility
6. startup/import cost
7. whether FrameVitals can implement the needed behavior with an existing dependency instead

A large dependency should not enter the base install merely because one analysis can use it.
