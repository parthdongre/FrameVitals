# FrameVitals

**Data-health diagnostics, drift detection, and ML-readiness checks for tabular data.**

[![Tests](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml/badge.svg)](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FrameVitals is an open-source Python toolkit for checking whether tabular data is healthy enough for analysis and machine learning. It combines structural profiling, data-quality scoring, statistical diagnostics, anomaly detection, drift analysis, ML-readiness checks, target-aware analysis, and optional AI-assisted interpretation behind one package.

> Version: `0.1.0` — alpha  
> Package: `framevitals`  
> License: MIT

## Install

FrameVitals supports Python 3.11, 3.12, and 3.13.

```bash
pip install framevitals
```

Optional feature groups:

```bash
pip install "framevitals[ml]"   # XGBoost, LightGBM, PyOD, SHAP
pip install "framevitals[ai]"   # Ollama client
pip install "framevitals[web]"  # Flask web API runtime
pip install "framevitals[all]"  # all optional runtime features
```

For development from source:

```bash
git clone https://github.com/parthdongre/FrameVitals.git
cd FrameVitals
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Quick start

FrameVitals accepts pandas DataFrames directly.

```python
import pandas as pd
import framevitals as fv

_df = pd.read_csv("customers.csv")
report = fv.analyze(_df)

print(report["health"]["overall_score"])
print(report["ml_readiness"])
```

File paths work too:

```python
report = fv.analyze("customers.csv", mode="quick")
```

With a supervised target:

```python
report = fv.analyze(
    _df,
    target="churn",
    mode="deep",
)

print(report["model_leaderboard"])
print(report["explainability"])
```

### Compare datasets for drift

```python
reference = pd.read_csv("train.csv")
current = pd.read_csv("production_batch.csv")

drift = fv.compare(reference, current)

print(drift["summary"]["overall_verdict"])
print(drift["columns"][:3])
```

Numeric columns use PSI, KS statistics, and standardized mean shift. Categorical columns use PSI and chi-square diagnostics.

### Filesystem artifacts are opt-in

Reusable Python calls do not write cleaned datasets or charts unless requested.

```python
report = fv.analyze(_df)
assert report["cleaning"]["output_path"] is None

report = fv.analyze(_df, artifacts=True)
print(report["cleaning"]["output_path"])
```

## CLI

```bash
framevitals --version
framevitals analyze dataset.csv
framevitals analyze dataset.csv --mode quick
framevitals analyze dataset.csv --target churn --mode deep
framevitals analyze dataset.csv --artifacts
framevitals analyze dataset.csv --output summary.json

framevitals compare train.csv production.csv
framevitals compare train.csv production.csv --columns age,income
framevitals compare train.csv production.csv --output drift.json
```

## Analysis modes

| Mode | Intended use |
| --- | --- |
| `quick` | Fast structural, quality, and ML-readiness checks |
| `standard` | Default deeper diagnostics |
| `deep` | Broader statistical and target-aware analysis |
| `research` | Largest analysis budget |

## Core capabilities

- pandas DataFrame plus CSV, TSV, Excel, and JSON inputs
- structural profiling and semantic column-role inference
- data-health and ML-readiness scoring
- missingness, duplicate, cardinality, and quality diagnostics
- deep statistical analysis
- ensemble anomaly detection
- reference-vs-current dataset drift comparison
- target leakage and multicollinearity diagnostics
- time-series and text-column profiling
- target-aware baseline/model leaderboard
- explainability with deterministic fallback when SHAP is unavailable
- conservative cleaning and optional chart/artifact generation
- JSON-friendly structured results
- Python API and CLI

Optional integrations add XGBoost, LightGBM, PyOD detectors, SHAP, Ollama-backed AI, and the Flask/React application stack.

## Optional web dashboard

The reusable library lives under `src/framevitals/`. The repository also includes an optional Flask API and React dashboard for interactive use.

Install the web dependencies:

```bash
pip install -e ".[web]"
```

Start the Flask API:

```bash
python app.py
```

Then, in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Typical local endpoints:

- Flask API: `http://127.0.0.1:5055`
- React dashboard: `http://127.0.0.1:5173`

## Project layout

```text
.
├── src/framevitals/          # Canonical installable Python package
├── tests/                    # Automated tests
├── frontend/                 # Optional React + TypeScript dashboard
├── templates/                # Optional Flask report pages
├── static/                   # Optional web/report assets
├── app.py                    # Optional Flask API/server
├── pyproject.toml            # Package metadata and dependency groups
└── .github/workflows/        # CI, packaging, and publishing
```

New reusable Python code should live in `src/framevitals/` and import through the `framevitals.*` namespace.

## Development

```bash
pip install -e ".[all,dev]"
pytest
python -m build
python -m twine check dist/*
```

CI tests the core installation across Python 3.11–3.13, separately smoke-tests optional features, validates dependency consistency, builds the React dashboard, checks the wheel contents, and installs the built wheel in a clean environment.

Development happens on `dev`; `main` is kept release-ready. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Releases

FrameVitals publishes through PyPI Trusted Publishing from GitHub Releases. See [RELEASING.md](RELEASING.md) for the release checklist.

## Roadmap

Near-term priorities include:

- FrameVitals Contracts and CI-friendly validation gates
- adaptive analysis execution based on dataset signals and resource budgets
- unified target intelligence for leakage, imbalance, redundancy, diagnostics, and explainability
- reusable baseline snapshots for monitoring drift and schema changes
- stable result/report objects with JSON and HTML exports
- configurable thresholds and custom checks
- deterministic large-dataset sampling and benchmark suites

## Contributing

Issues and pull requests are welcome. Keep changes focused, include tests for behavioral changes, and do not commit generated reports, datasets, credentials, caches, virtual environments, or build output.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License

FrameVitals is released under the [MIT License](LICENSE).
