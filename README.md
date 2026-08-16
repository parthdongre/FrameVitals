<div align="center">

# FrameVitals

### Know if your data is healthy, stable, and ML-ready — before your model finds out.

**A source-aware Python toolkit for data health, drift, contracts, quality gates, anomaly analysis, and ML-readiness diagnostics on tabular data.**

[![Tests](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml/badge.svg)](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/framevitals.svg)](https://pypi.org/project/framevitals/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/parthdongre/FrameVitals?style=social)](https://github.com/parthdongre/FrameVitals)

[Install](#installation) · [Quick start](#quick-start) · [Quality gates](#quality-gates) · [Sources](#source-aware-execution) · [CLI](#command-line-interface) · [Roadmap](#roadmap) · [Contributing](#contributing)

</div>

---

FrameVitals turns a supported tabular source into a **structured health report** that can be inspected, serialized, compared, snapshotted, extended with domain rules, and enforced in CI.

```python
import framevitals as fv

report = fv.analyze(data)
drift = fv.compare(reference, current)
contract = fv.infer_contract(reference)
validation = fv.validate(current, contract)
gate = fv.gate(current, reference=reference, contract=contract)
```

The goal is simple: **catch bad data before it becomes a bad model, a broken dashboard, or a production incident.**

```text
DataFrame / Arrow / files / DuckDB relation
                    │
                    ▼
              DatasetSource
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     bounded stream       exact/full path
          │                   │
          └─────────┬─────────┘
                    ▼
                 ANALYZE
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
   health        contract      snapshot
                    │             │
            ┌───────┘             ▼
            ▼                  history
         validate                 │
            │                     │
reference ─► compare              │
            │                     │
custom ─────┤                     │
checks      ▼                     │
           GATE ◄─────────────────┘
      PASS / WARN / FAIL
```

## Why FrameVitals?

| Question | FrameVitals |
| --- | --- |
| Is this dataset structurally healthy? | Missingness, duplicates, cardinality, schema and quality diagnostics |
| Is it ready for modelling? | ML-readiness scoring, target-aware checks and model diagnostics |
| Are there suspicious rows or features? | Statistical diagnostics, anomaly detection, leakage and multicollinearity checks |
| Has production data changed? | Numeric, categorical and schema drift diagnostics |
| Can I enforce expectations? | Versioned contracts, exact validation, custom checks and one quality gate |
| Can I monitor change without storing every raw batch? | Compact snapshots and persistent snapshot history |
| Will large supported sources be loaded into pandas by default? | Bounded source-aware execution where semantics permit it |
| Can I see when FrameVitals sampled or materialized data? | Execution provenance is included in source-aware results |
| Will analysis unexpectedly write files? | No — filesystem artifacts are opt-in |

FrameVitals is **package-first**. Product logic lives under `src/framevitals/`; the CLI, Flask API, React dashboard and reusable GitHub Action wrap the same canonical engines.

## Installation

FrameVitals supports **Python 3.11, 3.12, and 3.13**. The current public release is **0.1.0 (alpha)**.

```bash
pip install framevitals
```

Optional capabilities are split into extras so the base data-health engine stays focused:

```bash
pip install "framevitals[arrow]"   # Parquet + Arrow-backed source interoperability
pip install "framevitals[duckdb]"  # lazy DuckDB relations + Arrow transport
pip install "framevitals[excel]"   # XLS/XLSX readers
pip install "framevitals[plot]"    # Matplotlib/Seaborn charts and report plotting
pip install "framevitals[ml]"      # XGBoost, LightGBM, PyOD, SHAP
pip install "framevitals[ai]"      # Ollama-backed AI features
pip install "framevitals[web]"     # Flask web runtime
pip install "framevitals[all]"     # all optional runtime capabilities
```

On a development checkout:

```bash
pip install -e ".[all,dev]"
```

`arrow` enables projection-aware Parquet execution, compatible CSV/TSV streaming, native PyArrow table inputs, and Arrow PyCapsule-compatible table producers. `duckdb` adds lazy `DuckDBPyRelation` inputs without putting DuckDB in the base dependency set.

## Quick start

### Analyze a DataFrame or file

```python
import pandas as pd
import framevitals as fv

customers = pd.read_csv("customers.csv")
report = fv.analyze(customers)

print(report.health["overall_score"])
print(report.ml_readiness)
print(report.findings[:3])
```

File paths work directly:

```python
report = fv.analyze("customers.parquet", mode="quick")
```

### Analyze Arrow-native data

With the `arrow` extra installed:

```python
import pyarrow as pa
import framevitals as fv

table = pa.table({
    "age": [21, 34, 48],
    "income": [30_000, 62_000, 81_000],
})

report = fv.analyze(table, mode="quick")
profile = fv.profile(table)
```

PyArrow `Table` and `RecordBatch` inputs enter the source-aware batch path instead of being converted to pandas before analysis. Table-like objects implementing the Arrow C Stream / PyCapsule protocol can use the same interoperability boundary when PyArrow is installed.

### Analyze a lazy DuckDB relation

With the `duckdb` extra installed:

```python
import duckdb
import framevitals as fv

con = duckdb.connect()
orders = con.sql("""
    SELECT *
    FROM read_parquet('orders/*.parquet')
""")

profile = fv.profile(orders)
report = fv.analyze(orders, mode="quick")
```

FrameVitals obtains exact relation shape metadata, pushes requested column projection into the relation, and consumes Arrow record batches. It does not call `.df()` for diagnostics that can stay on the streaming path.

### Run only the diagnostic you need

```python
profile = fv.profile("customers.parquet")
health = fv.health("customers.parquet")
readiness = fv.ml_readiness("customers.parquet")
quality = fv.quality("customers.parquet")
stats = fv.statistics("customers.parquet", mode="quick")
anomalies = fv.anomalies("customers.parquet", mode="quick")
relationships = fv.relationships("customers.parquet")
```

Focused APIs avoid running unrelated pipeline stages. Supported streaming sources use bounded execution where the diagnostic permits it and disclose exact, sampled, estimated, or materialized behavior in result metadata.

### Add a supervised-learning target

```python
report = fv.analyze(
    customers,
    target="churn",
    mode="deep",
)

print(report["model_leaderboard"])
print(report["explainability"])
```

Target-aware analysis can surface modelling risks such as leakage, imbalance, redundant features, unstable relationships, and weak baselines.

## Drift and contracts

### Compare datasets for drift

```python
result = fv.compare(reference, current)

print(result.severity)
print(result["columns"][:3])
print(result["execution"])
```

Numeric drift uses PSI, Kolmogorov-Smirnov statistics and standardized mean shift. Categorical drift uses PSI and chi-square diagnostics. Streaming-capable sources can bound distribution work while preserving exact source-shape metadata.

### Infer and validate a contract

```python
contract = fv.infer_contract(reference)
result = fv.validate(current, contract)

if result.status == "fail":
    for finding in result.findings:
        print(finding["message"])
```

Contracts are versioned JSON-friendly dictionaries. Current inference can capture required/optional columns, broad data types, nullability, tolerated null rates, finite numeric bounds, low-cardinality allowed values, and uniqueness expectations.

**Validation intentionally remains exact.** Constraints such as uniqueness, allowed values and bounds are not silently downgraded to sampled approximations. For a non-pandas source, exact validation may therefore materialize a complete pandas representation, and the `execution.full_materialization` field reports that decision.

## Quality gates

`fv.gate(...)` combines the checks you choose into one CI-friendly verdict:

```python
result = fv.gate(
    current,
    reference=reference,
    contract=contract,
    drift_warn_on="moderate",
    drift_fail_on="severe",
)

print(result.status)   # pass / warn / fail
print(result.passed)   # True unless the gate failed
```

You can run only the families you need:

```python
fv.gate(current, contract=contract)
fv.gate(current, reference=reference)
fv.gate(current, reference=reference, contract=contract)
```

### Domain-specific custom checks

FrameVitals supports exact application-defined invariants without requiring a fork:

```python
@fv.check(
    "positive revenue",
    severity="error",
    description="Revenue cannot be negative.",
)
def positive_revenue(df):
    return {
        "passed": bool((df["revenue"] >= 0).all()),
        "message": "Negative revenue records were found.",
    }

checks = fv.run_checks(current, [positive_revenue])
gate = fv.gate(current, custom_checks=[positive_revenue])
```

`fv.run_checks()` returns a dict-compatible `CheckResult`. Arbitrary Python checks run against the complete DataFrame because FrameVitals cannot safely infer whether a user-defined invariant is sampleable. Non-pandas sources therefore report full materialization for this exact path.

### Third-party check plugins

Installed packages can register checks through Python entry points under `framevitals.checks`:

```toml
[project.entry-points."framevitals.checks"]
positive_revenue = "acme_data_checks:positive_revenue"
```

Discovery is deliberately **opt-in** because loading an entry point executes provider code:

```python
checks = fv.discover_checks()
result = fv.gate(current, custom_checks=checks)
```

FrameVitals does not import or execute installed check plugins automatically.

### GitHub Actions

The repository ships a reusable composite action at `action.yml`:

```yaml
- uses: parthdongre/FrameVitals@main
  id: framevitals
  with:
    current: data/production.parquet
    reference: data/training.parquet
    contract: data/contract.json
    drift-warn-on: moderate
    drift-fail-on: severe
    output: framevitals-gate.json

- name: Show verdict
  run: echo "FrameVitals status: ${{ steps.framevitals.outputs.status }}"
```

For production workflows, pin the action to a released tag or commit rather than a moving branch. The action exposes `status`, `passed`, and `result-path` outputs.

## Snapshots and monitoring history

An `AnalysisResult` can be reduced to a compact versioned snapshot:

```python
report = fv.analyze(current)
snapshot = report.snapshot("snapshot.json")
```

Snapshots retain a fingerprint and compact state such as schema, missingness, health, ML-readiness and finding codes without embedding the full raw dataset.

```python
previous = fv.load_snapshot("previous.json")
latest = fv.load_snapshot("snapshot.json")
change = fv.compare_snapshots(previous, latest)
```

For repeated local monitoring, use `SnapshotHistory`:

```python
history = fv.SnapshotHistory(".framevitals/history")

history.add(report.snapshot(), label="production")
latest = history.latest()
previous = history.previous()
change = history.compare_latest()
```

The default `.framevitals/` runtime directory is ignored by Git.

## The public API

FrameVitals keeps workflow entry points at the package root while retaining focused diagnostics for callers that need one answer rather than the complete pipeline.

| API | Purpose |
| --- | --- |
| `framevitals.analyze(...)` | Run the configured source-aware analysis workflow |
| `framevitals.plan(...)` | Preview applicable modules and execution constraints |
| `framevitals.profile(...)` | Profile structure, missingness and summaries only |
| `framevitals.roles(...)` | Infer semantic and structural column roles |
| `framevitals.health(...)` | Calculate data-health diagnostics only |
| `framevitals.ml_readiness(...)` | Calculate ML-readiness diagnostics only |
| `framevitals.quality(...)` | Run deterministic data-quality checks only |
| `framevitals.statistics(...)` | Run bounded deep statistical diagnostics |
| `framevitals.anomalies(...)` | Run bounded anomaly diagnostics |
| `framevitals.relationships(...)` | Discover strong numeric relationships |
| `framevitals.target_analysis(...)` | Run target-aware diagnostics only |
| `framevitals.compare(...)` | Compare reference/current data for drift |
| `framevitals.infer_contract(...)` | Infer a reusable versioned data contract |
| `framevitals.validate(...)` | Validate data against a contract |
| `framevitals.check(...)` | Define a reusable custom invariant |
| `framevitals.run_checks(...)` | Run exact custom checks and return `CheckResult` |
| `framevitals.discover_checks(...)` | Explicitly discover installed check plugins |
| `framevitals.gate(...)` | Combine contract, custom and drift checks into one verdict |
| `framevitals.create_snapshot(...)` | Create compact monitoring state from an analysis result |
| `framevitals.compare_snapshots(...)` | Compare two stored analysis states |
| `framevitals.SnapshotHistory(...)` | Persist and compare a compact local monitoring timeline |

`framevitals.api` remains a lazy compatibility facade over these canonical engines; it does not maintain a second implementation.

## What FrameVitals checks

| Area | Examples |
| --- | --- |
| **Structure** | shape, dtypes, semantic column roles, date/text detection |
| **Data quality** | missingness, duplicates, constants, cardinality, outliers |
| **Health scoring** | overall dataset health plus component diagnostics |
| **ML readiness** | modelling readiness, risky columns, preprocessing recommendations |
| **Statistics** | distribution checks, normality, correlations, effect-size style diagnostics |
| **Anomalies** | multivariate and robust outlier detectors, optional ensemble methods |
| **Target intelligence** | task inference, leakage hints, multicollinearity, feature/model diagnostics |
| **Drift** | PSI, KS, chi-square, mean shift, new/disappearing categories |
| **Contracts** | schema, types, nullability, bounds, domains and uniqueness expectations |
| **Custom invariants** | exact application-specific DataFrame predicates |
| **Time series** | date-aware diagnostics, stationarity, decomposition and forecast previews |
| **Text** | text-column profiling, vocabulary and lightweight semantic diagnostics |
| **Explainability** | model feature importance and SHAP when optional capabilities are installed |

Not every analysis runs on every dataset. FrameVitals uses dataset signals, selected mode, target availability, source capabilities, execution budgets and installed optional dependencies to decide what is useful and safe to execute.

## Analysis modes

```python
fv.analyze(data, mode="quick")
fv.analyze(data, mode="standard")
fv.analyze(data, mode="deep")
fv.analyze(data, mode="research")
```

| Mode | Best for |
| --- | --- |
| `quick` | Fast structural, quality and ML-readiness checks |
| `standard` | Everyday analysis with broader diagnostics |
| `deep` | Target-aware and heavier statistical analysis |
| `research` | Largest analysis budget for exploratory work |

## Source-aware execution

FrameVitals separates a **dataset source** from the diagnostic that consumes it. This lets supported inputs expose exact metadata, projection and record batches without forcing every public API to begin with a full pandas conversion.

```text
Pandas DataFrame
PyArrow Table / RecordBatch
Arrow C Stream / PyCapsule-compatible table
CSV / TSV / Parquet
DuckDB relation
        │
        ▼
   DatasetSource
        │
 ┌──────┴─────────┐
 ▼                ▼
stream/bounded   exact/full
 ▼                ▼
execution provenance
```

Current source behavior includes:

- **Pandas** — already materialized, canonical in-memory baseline.
- **Parquet** — Arrow-backed metadata, projection and record batches.
- **CSV/TSV** — Arrow streaming when compatible and the `arrow` extra is installed; pandas fallback otherwise.
- **PyArrow Table/RecordBatch** — native in-memory Arrow batch path.
- **Arrow-compatible table producers** — normalized through the standard Arrow C Stream / PyCapsule boundary when available.
- **DuckDB relation** — exact count/schema metadata, projection pushed into DuckDB, Arrow batch transport, no `.df()` on streaming-safe diagnostics.
- **Other supported files** — materialized through the established loader when no streaming adapter exists.

A source being streamable does **not** mean every operation is approximate. FrameVitals chooses execution semantics by diagnostic:

- source shape/schema can remain exact;
- some statistics, anomaly and drift work can operate on explicit bounded samples;
- exact contracts and arbitrary custom checks remain full-data operations;
- every source-aware result should disclose whether data was streamed, sampled, estimated or fully materialized.

## Filesystem artifacts are opt-in

Calling the Python API does not need to scatter reports and cleaned files around the working directory.

```python
report = fv.analyze(data)
assert report["cleaning"]["output_path"] is None

report = fv.analyze(data, artifacts=True)
print(report["cleaning"]["output_path"])
```

## Command-line interface

FrameVitals ships with a CLI for terminals, scripts and CI workflows.

```bash
framevitals --help
framevitals analyze --help
framevitals compare --help
framevitals gate --help
framevitals --version
```

Analyze and plan:

```bash
framevitals analyze dataset.csv
framevitals analyze dataset.csv --mode quick
framevitals analyze dataset.csv --target churn --mode deep
framevitals analyze dataset.csv --output report.json
framevitals plan dataset.csv --mode standard
```

Compare, contract and gate:

```bash
framevitals compare train.csv production.csv --fail-on moderate
framevitals infer-contract training_data.csv --output contract.json
framevitals validate production_batch.csv --contract contract.json
framevitals gate production_batch.csv \
  --reference training_data.csv \
  --contract contract.json \
  --output gate.json
```

CLI exit behavior is intentional:

- `framevitals validate`: `0` for pass/warn by default, `1` for warning when `--fail-on-warn` is enabled, `2` for contract failure.
- `framevitals compare`: `1` only when `--fail-on` is supplied and the configured drift severity is reached; otherwise `0`.
- `framevitals gate`: `0` for pass/warn and `1` for fail.

## Result objects

The main workflows return dict-compatible result objects so existing mapping-style code keeps working while application/notebook ergonomics improve:

- `AnalysisResult`
- `DriftResult`
- `ValidationResult`
- `CheckResult`
- `GateResult`
- `AnalysisSnapshot`

Example:

```python
report = fv.analyze(data)
report.health
report.findings
report.column("age")
report.to_json("report.json")
report.to_html("report.html")
report.snapshot("snapshot.json")
```

The `0.x` series preserves room to harden result schemas before `1.0` stability guarantees.

## Optional capabilities

The default package contains the core data-health engine. Additional integrations are imported only when their feature is requested.

```bash
pip install "framevitals[arrow]"
```

Adds Arrow-backed files and in-memory Arrow interoperability.

```bash
pip install "framevitals[duckdb]"
```

Adds lazy DuckDB relation support plus the Arrow transport required by that adapter.

```bash
pip install "framevitals[ml]"
```

Adds heavier model integrations including XGBoost, LightGBM, PyOD and SHAP.

```bash
pip install "framevitals[ai]"
```

Adds Ollama-backed interpretation and question-answering features. AI remains an optional explanation layer; computed diagnostics remain usable without it.

```bash
pip install "framevitals[plot]"
```

Adds Matplotlib/Seaborn-backed chart rendering and report plotting. Structured diagnostics do not require this extra.

```bash
pip install "framevitals[excel]"
```

Adds XLS/XLSX reader engines. CSV/TSV/JSON workflows remain available without them.

## Web dashboard

The repository includes an optional **Flask API + React/TypeScript dashboard** for interactive exploration.

```bash
pip install -e ".[web]"
python app.py
```

Then in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

The `web` extra contains the server runtime only. Install `plot` for server-side chart/report artifacts and `ai` for the agentic Q&A path. Optional stacks are loaded lazily rather than blocking Flask startup.

Typical local endpoints:

- Flask API: `http://127.0.0.1:5055`
- React dashboard: `http://127.0.0.1:5173`

## Design principles

- **Package first** — Python APIs are canonical; interfaces wrap them.
- **Structured results** — return reusable data, not only screenshots or prose.
- **Source aware** — inspect and stream supported inputs before choosing materialization.
- **Bounded execution** — expensive diagnostics operate within explicit scale budgets.
- **Transparent approximations** — sampled or estimated results disclose provenance.
- **Exact where correctness requires it** — contracts and arbitrary custom checks are not silently weakened.
- **Safe defaults** — no unexpected artifact writes or automatic plugin execution.
- **Small workflow surface** — analyze → compare → validate → gate → monitor stays obvious.
- **Optional integrations** — Arrow, DuckDB, Excel, plotting, ML, AI and web are independently installable.
- **Extensible without forks** — domain checks and opt-in entry-point plugins can extend the gate.

## Project layout

```text
.
├── src/framevitals/          # canonical installable Python package
├── tests/                    # automated test suite
├── benchmarks/               # scale/performance harnesses
├── rust/                     # optional native acceleration core
├── frontend/                 # optional React + TypeScript dashboard
├── templates/                # Flask report pages
├── static/                   # web/report assets
├── examples/                 # focused usage examples
├── action.yml                # reusable GitHub Actions quality gate
├── app.py                    # optional Flask API/server
├── pyproject.toml            # package metadata and dependency groups
└── .github/workflows/        # CI, interop, package, benchmark and release workflows
```

New reusable Python code belongs in `src/framevitals/` and should import through the `framevitals.*` namespace.

## Development

```bash
git clone https://github.com/parthdongre/FrameVitals.git
cd FrameVitals
git switch dev

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all,dev]"

pytest
python -m build
python -m twine check dist/*
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

CI validates the core package across Python 3.11–3.13, Arrow streaming paths, Arrow/DuckDB interoperability, optional dependency boundaries, the reusable Gate Action, the Rust workspace/native bridge, React build, wheel contents, distribution metadata, release-version consistency, and clean-wheel installation. A separate benchmark workflow records reproducible time and peak-RSS measurements.

Development is integrated through `dev`; `main` is kept release-ready.

## Roadmap

FrameVitals is moving toward a dependable data-health quality gate rather than an ever-growing collection of unrelated analytics modules.

```text
0.1  FOUNDATION
     analyze · focused diagnostics · drift · package/CLI baseline

0.2  SOURCE-AWARE QUALITY GATES
     bounded streaming · contracts · validate · gate · snapshots · provenance

0.3  STABLE RESULTS + PERFORMANCE
     result-schema hardening · regression budgets · lighter installs · richer history

0.4  EXTENSIBILITY + INTEROPERABILITY
     custom checks · entry-point plugins · source protocols · integrations

1.0  STABLE DATA-HEALTH API
     dependable analyze → compare → validate → gate → monitor workflow
```

Several post-0.1 capabilities are already under active development on repository development branches. Before `1.0`, naming, thresholds, result schemas and extension points may still evolve.

## Project status

FrameVitals `0.1.x` is **alpha software**. The project already has a package API, CLI, contracts, gates, snapshots, source-aware execution, tests and release tooling, while the `0.x` series deliberately leaves room to refine schemas, thresholds, dependency boundaries and extension points before stability guarantees begin.

Feedback on real datasets, false positives, missing diagnostics, performance, source compatibility and API ergonomics is especially valuable.

## Contributing

Contributions are welcome. A good contribution is focused, tested, and improves diagnostic reliability, public workflow clarity, source compatibility, execution transparency or extension ergonomics.

Start with [CONTRIBUTING.md](CONTRIBUTING.md), and read the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

## Releases

Releases are built and validated in GitHub Actions and published through PyPI Trusted Publishing. See [RELEASING.md](RELEASING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

FrameVitals is open source under the [MIT License](LICENSE).

---

<div align="center">

**If FrameVitals is useful to you, consider starring the repository — it helps the project grow.**

</div>
