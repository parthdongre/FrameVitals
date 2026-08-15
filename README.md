<div align="center">

# FrameVitals

### Know if your data is healthy, stable, and ML-ready — before your model finds out.

**A Python toolkit for data-quality diagnostics, drift detection, anomaly analysis, and ML-readiness checks on pandas and tabular data.**

[![Tests](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml/badge.svg)](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Release](https://img.shields.io/badge/release-0.1.0%20alpha-6f42c1)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/parthdongre/FrameVitals?style=social)](https://github.com/parthdongre/FrameVitals)

[Install](#installation) · [Quick start](#quick-start) · [CLI](#command-line-interface) · [Roadmap](#roadmap) · [Contributing](#contributing)

</div>

---

FrameVitals turns a pandas DataFrame or tabular dataset into a **structured health report** you can inspect, serialize, compare, and eventually enforce in CI.

Instead of stitching together separate profiling, quality, drift, anomaly, and ML-readiness tools, FrameVitals gives you one deliberately small entry point:

```python
import framevitals as fv

report = fv.analyze(df)
drift = fv.compare(reference_df, current_df)
```

The goal is simple: **catch bad data before it becomes a bad model, a broken dashboard, or a production incident.**

```text
                  ┌──────────────────────────┐
DataFrame / file ─►        ANALYZE           │
                  │ profile · health · ML    │
                  │ stats · anomalies · risk │
                  └────────────┬─────────────┘
                               │
                               ▼
                        structured report

Reference + current ───────────► COMPARE ─────► drift verdict
```

## Why FrameVitals?

Most data checks answer one narrow question. FrameVitals is designed around the questions that show up repeatedly in real data and ML workflows:

| Question | FrameVitals |
| --- | --- |
| Is this dataset structurally healthy? | Missingness, duplicates, cardinality, schema and quality diagnostics |
| Is it ready for modelling? | ML-readiness scoring, target-aware checks and model diagnostics |
| Are there suspicious rows or features? | Statistical diagnostics, anomaly detection, leakage and multicollinearity checks |
| Has production data changed? | Reference-vs-current drift analysis with numeric and categorical tests |
| Can I use the result in code? | JSON-friendly structured output through a Python API and CLI |
| Will analysis unexpectedly write files? | No — filesystem artifacts are opt-in |

FrameVitals is **package-first**. The core library lives under `src/framevitals/`; the Flask API and React dashboard are optional interfaces around the same analysis engine.

## Installation

FrameVitals supports **Python 3.11, 3.12, and 3.13**.

```bash
pip install framevitals
```

Optional feature groups keep heavier dependencies out of the default install:

```bash
pip install "framevitals[ml]"   # XGBoost, LightGBM, PyOD, SHAP
pip install "framevitals[ai]"   # Ollama-backed AI features
pip install "framevitals[web]"  # Flask web runtime
pip install "framevitals[all]"  # all optional runtime features
```

## Quick start

### Analyze a DataFrame

```python
import pandas as pd
import framevitals as fv

customers = pd.read_csv("customers.csv")
report = fv.analyze(customers)

print(report["health"]["overall_score"])
print(report["ml_readiness"])
```

File paths work too:

```python
report = fv.analyze("customers.csv", mode="quick")
```

FrameVitals supports pandas DataFrames and common tabular file formats including CSV, TSV, Excel, and JSON.

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

Numeric drift uses **PSI, Kolmogorov-Smirnov statistics, and standardized mean shift**. Categorical drift uses **PSI and chi-square diagnostics**.

## The public API

The public API is intentionally small while FrameVitals is in alpha.

| API | Status | Purpose |
| --- | --- | --- |
| `framevitals.analyze(...)` | Available in `0.1.0` | Profile and diagnose one dataset |
| `framevitals.compare(...)` | Available in `0.1.0` | Compare reference and current data for drift |
| `framevitals.validate(...)` | In development | Validate data against an inferred or explicit contract |
| snapshots / monitoring | Roadmap | Reuse baselines for recurring schema and drift checks |

This keeps the library easy to learn while leaving room for the result model and validation system to mature before `1.0`.

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
| **Time series** | date-aware diagnostics, stationarity, decomposition and forecast previews |
| **Text** | text-column profiling, vocabulary and lightweight semantic diagnostics |
| **Explainability** | model feature importance and SHAP when the optional ML stack is installed |

Not every analysis runs on every dataset. FrameVitals uses dataset signals, selected mode, target availability, and installed optional dependencies to decide what is useful and safe to execute.

## Analysis modes

```python
fv.analyze(df, mode="quick")
fv.analyze(df, mode="standard")
fv.analyze(df, mode="deep")
fv.analyze(df, mode="research")
```

| Mode | Best for |
| --- | --- |
| `quick` | Fast structural, quality, and ML-readiness checks |
| `standard` | Everyday analysis with broader diagnostics |
| `deep` | Target-aware and heavier statistical analysis |
| `research` | Largest analysis budget for exploratory work |

## Filesystem artifacts are opt-in

FrameVitals is designed to behave like a library first. Calling the Python API does not need to scatter reports and cleaned files around your working directory.

```python
report = fv.analyze(df)
assert report["cleaning"]["output_path"] is None

report = fv.analyze(df, artifacts=True)
print(report["cleaning"]["output_path"])
```

## Command-line interface

FrameVitals also ships with a CLI for scripts, terminals, and future CI workflows.

```bash
framevitals --version

framevitals analyze dataset.csv
framevitals analyze dataset.csv --mode quick
framevitals analyze dataset.csv --target churn --mode deep
framevitals analyze dataset.csv --output report.json
framevitals analyze dataset.csv --artifacts

framevitals compare train.csv production.csv
framevitals compare train.csv production.csv --columns age,income
framevitals compare train.csv production.csv --output drift.json
```

## Optional ML and AI features

The default package contains the core data-health engine. Heavier features are separated into extras so a simple install stays predictable.

```bash
pip install "framevitals[ml]"
```

Adds optional integrations including XGBoost, LightGBM, PyOD and SHAP.

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

The public project website will remain separate from the package runtime so the library does not depend on a hosted service.

## Design principles

FrameVitals is being built around a few constraints that are easy to lose in analytics projects:

- **DataFrame first** — use it directly from Python without routing through a web app.
- **Structured results** — return reusable data, not only screenshots or prose.
- **Safe defaults** — no unexpected artifact writes and graceful optional-feature fallbacks.
- **Small public API** — make the common path obvious before exposing every internal module.
- **Optional heavy dependencies** — ML, AI, and web features should not bloat a basic install.
- **Production direction** — drift, contracts, snapshots, and CI quality gates are first-class roadmap items.

## Project layout

```text
.
├── src/framevitals/          # canonical installable Python package
├── tests/                    # automated test suite
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

CI validates the core package across Python 3.11–3.13, optional features, the React build, wheel contents, distribution metadata, and a clean-wheel install.

Development is integrated through `dev`; `main` is kept release-ready.

## Roadmap

FrameVitals is moving toward a complete data-health quality gate:

```text
0.1  ANALYZE + COMPARE
     data health · ML readiness · target diagnostics · drift

0.2  VALIDATE + SNAPSHOTS
     data contracts · CI gates · reusable baselines

0.3  RESULT OBJECTS + ADVANCED DRIFT
     stronger result model · large-data handling · richer monitoring

0.4  EXTENSIBILITY + INTEGRATIONS
     configurable checks · adapters · monitoring workflows

1.0  STABLE DATA-HEALTH API
     dependable analyze → compare → validate → monitor workflow
```

Near-term work is tracked through issues and the `dev` branch.

## Project status

FrameVitals `0.1.x` is **alpha software**. The core API is usable, but the project is intentionally still refining naming, result schemas, thresholds, and extension points before `1.0`.

If you are using FrameVitals in a project, feedback about real datasets, false positives, missing diagnostics, performance, and API ergonomics is especially valuable.

## Contributing

Contributions are welcome.

A good contribution is focused, tested, and improves either the reliability of a diagnostic or the clarity of the public workflow.

Start with [CONTRIBUTING.md](CONTRIBUTING.md), and please read the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security Policy](SECURITY.md).

## Releases

Releases are built and validated in GitHub Actions and published through PyPI Trusted Publishing. See [RELEASING.md](RELEASING.md) and [CHANGELOG.md](CHANGELOG.md).

## License

FrameVitals is open source under the [MIT License](LICENSE).

---

<div align="center">

**If FrameVitals is useful to you, consider starring the repository — it helps the project grow.**

</div>
