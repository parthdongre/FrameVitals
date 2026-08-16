<div align="center">

# FrameVitals

### Know if your data is healthy, stable, and ML-ready — before your model finds out.

**A Python toolkit for data-health diagnostics, drift detection, contracts, quality gates, anomaly analysis, and ML-readiness checks on tabular data.**

[![Tests](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml/badge.svg)](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/framevitals.svg)](https://pypi.org/project/framevitals/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/parthdongre/FrameVitals?style=social)](https://github.com/parthdongre/FrameVitals)

[Install](#installation) · [Quick start](#quick-start) · [Quality gates](#quality-gates) · [CLI](#command-line-interface) · [Roadmap](#roadmap) · [Contributing](#contributing)

</div>

---

FrameVitals turns a pandas DataFrame or tabular dataset into a **structured health report** that can be inspected, serialized, compared, snapshotted, and enforced in CI.

Instead of stitching together separate profiling, quality, drift, anomaly, contract, and ML-readiness tools, FrameVitals exposes a compact workflow:

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
                         ┌──────────────────────────┐
DataFrame / file ───────►│         ANALYZE          │
                         │ profile · health · ML     │
                         │ stats · anomalies · risk  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                               structured result
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
                    snapshot                    contract
                        │                           │
Reference + current ────┴──► compare / validate ───┘
                                      │
                                      ▼
                                    GATE
                              PASS / WARN / FAIL
```

## Why FrameVitals?

Most data checks answer one narrow question. FrameVitals is designed around the questions that repeatedly show up in real data and ML workflows:

| Question | FrameVitals |
| --- | --- |
| Is this dataset structurally healthy? | Missingness, duplicates, cardinality, schema and quality diagnostics |
| Is it ready for modelling? | ML-readiness scoring, target-aware checks and model diagnostics |
| Are there suspicious rows or features? | Statistical diagnostics, anomaly detection, leakage and multicollinearity checks |
| Has production data changed? | Reference-vs-current numeric, categorical and schema drift diagnostics |
| Can I enforce expectations? | Versioned data contracts, validation and one CI-friendly quality gate |
| Can I track change without storing every raw dataset? | Compact versioned analysis snapshots and snapshot diffs |
| Will large Arrow-compatible files be fully loaded by default? | Source-aware bounded execution for supported streaming workflows |
| Will analysis unexpectedly write files? | No — filesystem artifacts are opt-in |

FrameVitals is **package-first**. The canonical library lives under `src/framevitals/`; the Flask API and React dashboard are optional interfaces around the same analysis engine.

## Installation

FrameVitals supports **Python 3.11, 3.12, and 3.13**. The current public release is **0.1.0 (alpha)** and is available on PyPI.

```bash
pip install framevitals
```

Optional capabilities are installed as extras:

```bash
pip install "framevitals[arrow]"  # bounded Parquet / compatible CSV / TSV streaming
pip install "framevitals[ml]"     # XGBoost, LightGBM, PyOD, SHAP
pip install "framevitals[ai]"     # Ollama-backed AI features
pip install "framevitals[web]"    # Flask web runtime
pip install "framevitals[all]"    # all optional runtime capabilities
```

Parquet streaming requires the `arrow` extra. With Arrow installed, compatible CSV and TSV inputs can also use the source-aware streaming engine; without Arrow they retain the normal pandas-backed file path.

## Quick start

### Analyze a DataFrame

```python
import pandas as pd
import framevitals as fv

customers = pd.read_csv("customers.csv")
report = fv.analyze(customers)

print(report.health["overall_score"])
print(report.ml_readiness)
print(report.findings[:3])
```

File paths work too:

```python
report = fv.analyze("customers.csv", mode="quick")
```

FrameVitals supports pandas DataFrames and common tabular file formats including CSV, TSV, Excel, JSON, and Parquet when the required reader capability is installed.

### Run only the diagnostic you need

The focused API avoids running the complete analysis pipeline when only one answer is required:

```python
profile = fv.profile("customers.parquet")
health = fv.health("customers.parquet")
readiness = fv.ml_readiness("customers.parquet")
stats = fv.statistics("customers.parquet", mode="quick")
anomalies = fv.anomalies("customers.parquet", mode="quick")
relationships = fv.relationships("customers.parquet")
```

Supported streaming sources use bounded execution where the diagnostic permits it and disclose execution provenance in their result metadata.

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

### Compare datasets for drift

```python
reference = pd.read_csv("training_data.csv")
current = pd.read_csv("production_batch.csv")

result = fv.compare(reference, current)

print(result["summary"]["overall_verdict"])
print(result["columns"][:3])
```

Numeric drift uses PSI, Kolmogorov-Smirnov statistics, and standardized mean shift. Categorical drift uses PSI and chi-square diagnostics.

For streaming-capable file sources, value-distribution work is bounded and the result reports the true source shapes plus the sampling strategy used for distribution diagnostics.

### Validate a data contract

Infer a contract once from a trusted reference dataset, then validate later batches before they reach downstream jobs:

```python
contract = fv.infer_contract(reference)
result = fv.validate(current, contract)

if result["status"] == "fail":
    for finding in result["errors"]:
        print(finding["message"])
```

Contracts are versioned JSON-friendly dictionaries. Current contract inference can capture required or optional columns, broad data types, nullability and tolerated null rates, finite numeric bounds, low-cardinality allowed values, and uniqueness expectations.

Validation intentionally remains exact for constraints such as uniqueness, allowed values and bounds rather than silently downgrading those checks to sampled approximations.

## Quality gates

`fv.gate(...)` combines contract validation and drift into one CI-friendly verdict:

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

You can run only the checks you need:

```python
fv.gate(current, contract=contract)
fv.gate(current, reference=reference)
fv.gate(current, reference=reference, contract=contract)
```

Contract validation remains exact. Drift can use bounded source-aware sampling where a streaming input supports it, and the returned `execution` block records that distinction.

## Snapshots

An `AnalysisResult` can be reduced to a compact versioned monitoring snapshot:

```python
report = fv.analyze(current)
snapshot = report.snapshot("snapshot.json")
```

Snapshots retain a fingerprint plus compact state such as schema, missingness, health, ML-readiness and finding codes without embedding the full raw dataset.

```python
previous = fv.load_snapshot("previous.json")
latest = fv.load_snapshot("snapshot.json")
change = fv.compare_snapshots(previous, latest)

print(change["schema"])
print(change["health_delta"])
print(change["findings"]["new"])
```

## The public API

FrameVitals keeps workflow entry points at the package root and exposes focused diagnostics for callers that do not need the full pipeline.

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
| `framevitals.gate(...)` | Combine validation and drift into one quality verdict |
| `framevitals.create_snapshot(...)` | Create compact monitoring state from an analysis result |
| `framevitals.compare_snapshots(...)` | Compare two stored analysis states |

The compatibility module `framevitals.api` delegates lazily to these same canonical engines so older imports do not maintain a second implementation.

## What FrameVitals checks

| Area | Examples |
| --- | --- |
| **Structure** | shape, dtypes, semantic column roles, date/text detection |
| **Data quality** | missingness, duplicates, constants, cardinality, outliers |
| **Health scoring** | overall dataset health plus component-level diagnostics |
| **ML readiness** | modelling readiness, risky columns, preprocessing recommendations |
| **Statistics** | distribution checks, normality, correlations, effect-size style diagnostics |
| **Anomalies** | multivariate and robust outlier detectors, optional ensemble methods |
| **Target intelligence** | task inference, leakage hints, multicollinearity, feature/model diagnostics |
| **Drift** | PSI, KS, chi-square, mean shift, new or disappearing categories |
| **Contracts** | schema, types, nullability, bounds, domains and uniqueness expectations |
| **Time series** | date-aware diagnostics, stationarity, decomposition and forecast previews |
| **Text** | text-column profiling, vocabulary and lightweight semantic diagnostics |
| **Explainability** | model feature importance and SHAP when the optional ML stack is installed |

Not every analysis runs on every dataset. FrameVitals uses dataset signals, selected mode, target availability, source capabilities, execution budgets, and installed optional dependencies to decide what is useful and safe to execute.

## Analysis modes

```python
fv.analyze(data, mode="quick")
fv.analyze(data, mode="standard")
fv.analyze(data, mode="deep")
fv.analyze(data, mode="research")
```

| Mode | Best for |
| --- | --- |
| `quick` | Fast structural, quality, and ML-readiness checks |
| `standard` | Everyday analysis with broader diagnostics |
| `deep` | Target-aware and heavier statistical analysis |
| `research` | Largest analysis budget for exploratory work |

## Source-aware execution

FrameVitals separates a dataset source from the diagnostic that consumes it. That lets supported file formats expose metadata, projection and record batches without forcing every public API to begin with a full pandas materialization.

```text
DataFrame / CSV / TSV / Parquet / ...
                  │
                  ▼
            DatasetSource
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   streaming path      materialized path
        │                   │
        └─────────┬─────────┘
                  ▼
         bounded diagnostics
```

Parquet is projection-aware and streaming through the Arrow capability. Compatible CSV/TSV sources can also use Arrow record batches. Focused diagnostics disclose when a result is exact, derived from a bounded row sample, or required full materialization.

## Filesystem artifacts are opt-in

FrameVitals is designed to behave like a library first. Calling the Python API does not need to scatter reports and cleaned files around the working directory.

```python
report = fv.analyze(data)
assert report["cleaning"]["output_path"] is None

report = fv.analyze(data, artifacts=True)
print(report["cleaning"]["output_path"])
```

## Command-line interface

FrameVitals ships with a CLI for scripts, terminals and CI workflows.

Discover commands and options:

```bash
framevitals --help
framevitals analyze --help
framevitals compare --help
framevitals gate --help
framevitals --version
```

Analyze a dataset:

```bash
framevitals analyze dataset.csv
framevitals analyze dataset.csv --mode quick
framevitals analyze dataset.csv --target churn --mode deep
framevitals analyze dataset.csv --output report.json
framevitals analyze dataset.csv --artifacts
```

Preview work without executing the heavy stages:

```bash
framevitals plan dataset.csv --mode standard
```

Compare two datasets for drift:

```bash
framevitals compare train.csv production.csv
framevitals compare train.csv production.csv --columns age,income
framevitals compare train.csv production.csv --fail-on moderate
framevitals compare train.csv production.csv --output drift.json
```

Create and validate a contract:

```bash
framevitals infer-contract training_data.csv --output contract.json
framevitals validate production_batch.csv --contract contract.json
```

Run the combined quality gate:

```bash
framevitals gate production_batch.csv \
  --reference training_data.csv \
  --contract contract.json
```

Tune CI thresholds when needed:

```bash
framevitals gate production_batch.csv \
  --reference training_data.csv \
  --contract contract.json \
  --drift-warn-on moderate \
  --drift-fail-on severe \
  --fail-on-validation-warning \
  --output gate.json
```

CLI exit behavior is intentional:

- `framevitals validate`: `0` for pass/warn by default, `1` for a warning when `--fail-on-warn` is enabled, and `2` for contract failure.
- `framevitals compare`: `1` only when `--fail-on` is supplied and the configured drift severity is reached; otherwise `0`.
- `framevitals gate`: `0` for pass/warn and `1` for fail.

## Result objects

`fv.analyze(...)` returns an `AnalysisResult`, a dict-compatible result object with notebook- and application-friendly helpers:

```python
report = fv.analyze(data)

report.health
report.ml_readiness
report.findings
report.recommendations
report.column("age")
report.summary()
report.to_json("report.json")
report.to_html("report.html")
report.snapshot("snapshot.json")
```

Drift, validation and gate workflows similarly return dict-compatible `DriftResult`, `ValidationResult` and `GateResult` objects with convenience properties while preserving JSON-friendly mapping behavior during the `0.x` series.

## Optional ML and AI features

The default package contains the core data-health engine. Additional integrations are separated into extras.

```bash
pip install "framevitals[ml]"
```

Adds integrations including XGBoost, LightGBM, PyOD and SHAP.

```bash
pip install "framevitals[ai]"
```

Adds Ollama-backed interpretation and question-answering features. AI is treated as an optional explanation layer; computed diagnostics remain usable without a reachable model.

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

Typical local endpoints:

- Flask API: `http://127.0.0.1:5055`
- React dashboard: `http://127.0.0.1:5173`

The public project website can remain separate from the package runtime so the Python library does not depend on a hosted service.

## Design principles

FrameVitals is being built around a few constraints that are easy to lose in analytics projects:

- **Package first** — the Python API is canonical; web interfaces wrap it rather than define it.
- **Structured results** — return reusable data, not only screenshots or prose.
- **Source aware** — inspect and stream supported sources instead of eagerly loading everything by default.
- **Bounded execution** — expensive diagnostics operate within explicit scale budgets.
- **Transparent approximations** — sampled or approximate results disclose their execution provenance.
- **Exact where correctness requires it** — contract constraints are not silently weakened to sampled checks.
- **Safe defaults** — no unexpected artifact writes and graceful optional-feature fallbacks.
- **Small workflow surface** — make analyze → compare → validate → gate obvious while keeping focused diagnostics available.
- **Optional integrations** — ML, AI, Arrow and web capabilities can be installed independently.

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
├── app.py                    # optional Flask API/server
├── pyproject.toml            # package metadata and dependency groups
└── .github/workflows/        # CI, package validation and publishing
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

CI validates the core package across Python 3.11–3.13, Arrow streaming paths, optional features, the Rust workspace/native bridge, React build, wheel contents, distribution metadata, and a clean-wheel install.

Development is integrated through `dev`; `main` is kept release-ready.

## Roadmap

FrameVitals is moving toward a dependable data-health quality gate rather than an ever-growing collection of unrelated analytics modules.

```text
0.1  FOUNDATION
     analyze · focused diagnostics · drift · package/CLI baseline

0.2  SOURCE-AWARE QUALITY GATES
     bounded streaming · contracts · validate · gate · snapshots · execution provenance

0.3  STABLE RESULTS + PERFORMANCE
     result-schema hardening · performance regression CI · lighter installs · richer history

0.4  EXTENSIBILITY + DATA ADAPTERS
     custom checks · plugin surface · Arrow/Polars/DuckDB-style adapters · integrations

1.0  STABLE DATA-HEALTH API
     dependable analyze → compare → validate → gate → monitor workflow
```

Several `0.2` capabilities are already under active development on the repository development branches. Before `1.0`, naming, thresholds, result schemas and extension points may still evolve.

## Project status

FrameVitals `0.1.x` is **alpha software**. The project already has a usable package API, CLI, contracts, gates, snapshots, source-aware execution, tests and release tooling, but the `0.x` series deliberately leaves room to refine result schemas, thresholds, dependency boundaries and extension points before stability guarantees begin.

Feedback on real datasets, false positives, missing diagnostics, performance, source compatibility and API ergonomics is especially valuable.

## Contributing

Contributions are welcome.

A good contribution is focused, tested, and improves either the reliability of a diagnostic, the clarity of the public workflow, source compatibility, or execution transparency.

Start with [CONTRIBUTING.md](CONTRIBUTING.md), and please read the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

## Releases

Releases are built and validated in GitHub Actions and published through PyPI Trusted Publishing. See [RELEASING.md](RELEASING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

FrameVitals is open source under the [MIT License](LICENSE).

---

<div align="center">

**If FrameVitals is useful to you, consider starring the repository — it helps the project grow.**

</div>
