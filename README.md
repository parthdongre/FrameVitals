# FrameVitals

**Data-health diagnostics, drift detection, and ML-readiness checks for tabular data.**

FrameVitals is an open-source Python toolkit for inspecting tabular datasets before they reach a model or production pipeline. It combines structural profiling, quality scoring, statistical diagnostics, anomaly detection, drift and time-series analysis, ML-readiness checks, target-aware modeling, explainability, cleaning, visualization, and optional AI-assisted interpretation behind one package.

> Package: `framevitals`  
> Status: `0.1.0` — alpha

## Install

FrameVitals supports Python 3.11, 3.12, and 3.13.

```bash
pip install framevitals
```

Optional feature groups are available when you need the heavier integrations:

```bash
pip install "framevitals[ml]"   # XGBoost, LightGBM, PyOD, SHAP
pip install "framevitals[ai]"   # Ollama client
pip install "framevitals[web]"  # Flask, Streamlit, Gunicorn
pip install "framevitals[all]"  # every optional runtime feature
```

The core engine still works without the optional ML libraries: XGBoost, LightGBM, PyOD, and SHAP are detected lazily and their analyses fall back or skip cleanly when unavailable.

For development from source:

```bash
git clone https://github.com/parthdongre/FrameVitals.git
cd FrameVitals
git switch dev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Python API

FrameVitals accepts both in-memory pandas DataFrames and supported dataset files.

```python
import pandas as pd
import framevitals as fv

# Analyze data already in memory.
df = pd.read_csv("customers.csv")
report = fv.analyze(df, mode="standard")

print(report["health"]["overall_score"])
print(report["ml_readiness"])
```

File paths work directly too:

```python
report = fv.analyze("customers.csv", mode="quick")
```

With a supervised target:

```python
report = fv.analyze(
    df,
    target="churn",
    mode="deep",
)

print(report["model_leaderboard"])
print(report["explainability"])
```

### Compare datasets for drift

Use a known reference dataset as a baseline and compare newer data against it:

```python
reference = pd.read_csv("train.csv")
current = pd.read_csv("production_batch.csv")

drift = fv.compare(reference, current)

print(drift["summary"]["overall_verdict"])
print(drift["columns"][:3])
```

Reference/current inputs can independently be DataFrames or file paths. Numeric columns report PSI, KS statistics, and standardized mean shift; categorical columns report PSI and chi-square diagnostics.

### Filesystem artifacts are opt-in

Reusable Python calls do not write cleaned datasets or charts unless requested:

```python
report = fv.analyze(df, mode="standard")
assert report["cleaning"]["output_path"] is None

report = fv.analyze(df, mode="standard", artifacts=True)
print(report["cleaning"]["output_path"])
```

The Flask/Streamlit application layer still enables its artifact workflow explicitly.

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

`framevitals --version` and importing the top-level package are intentionally lightweight; the analytics pipeline is loaded only when an analysis is requested.

## Analysis modes

| Mode | Intended use |
| --- | --- |
| `quick` | Fast structural, quality, and ML-readiness checks |
| `standard` | Default deeper diagnostics |
| `deep` | Broader statistical and target-aware analysis |
| `research` | Largest analysis budget |

## Core capabilities

- pandas DataFrame plus CSV, TSV, Excel, and JSON inputs
- Structural profiling and semantic column-role inference
- Data-health and ML-readiness scoring
- Missingness, duplicate, cardinality, and quality diagnostics
- Deep statistical analysis
- Ensemble anomaly detection
- Reference-vs-current dataset drift comparison
- Dedicated target-leakage and multicollinearity diagnostics
- Time-series and text-column profiling
- Target-aware model leaderboard
- Explainability with deterministic fallback when SHAP is unavailable
- Conservative cleaning and optional chart/artifact generation
- JSON-friendly structured results
- Python API and CLI

Optional integrations add XGBoost, LightGBM, PyOD detectors, SHAP plots, Ollama-backed AI, and the Flask/React/Streamlit application stack.

## Web development stack

For contributors who want the existing application interfaces as well as the package:

```bash
./install.sh
./run.sh
```

Typical local endpoints:

- Flask API: `http://127.0.0.1:5055`
- React dashboard: `http://127.0.0.1:5173`
- Streamlit console: `http://127.0.0.1:8501`

The application is an interface around FrameVitals. `src/framevitals/` is the canonical reusable library.

## Project layout

```text
.
├── src/framevitals/          # Canonical installable Python package
├── tests/                    # Automated tests
├── frontend/                 # Optional React + TypeScript dashboard
├── modules/                  # Temporary compatibility shims
├── demo_datasets/            # Example datasets
├── app.py                    # Optional Flask application
├── streamlit_app.py          # Optional Streamlit console
├── pyproject.toml            # Package metadata and dependency groups
└── .github/workflows/        # Tests, package validation, publishing
```

New reusable Python code should import through `framevitals.*`, never `modules.*`.

## Development

Ongoing development happens on `dev`; `main` is kept release-ready.

```bash
git switch dev
pip install -e ".[all,dev]"
pytest
python -m build
python -m twine check dist/*
```

CI tests the core installation across Python 3.11–3.13, separately smoke-tests the optional feature set, validates dependency consistency, and installs the built wheel in a clean virtual environment.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Packaging and releases

The distribution is built from the `src/` layout and explicitly excludes the legacy `modules/` compatibility namespace. GitHub Actions validates the wheel on `dev`, `main`, and pull requests.

Release publishing is configured for PyPI Trusted Publishing. See [RELEASING.md](RELEASING.md) for the one-time PyPI/GitHub environment setup and release checklist.

## Product roadmap

Near-term product priorities:

- Introduce **FrameVitals Contracts**: infer data-health expectations from a reference dataset and validate future data with pass/warn/fail results.
- Add CI-friendly `framevitals validate` / `framevitals check` commands with meaningful exit codes.
- Turn analysis results into a stable report/result object with JSON and HTML export methods.
- Make the signal-driven analysis selector actually control which expensive diagnostics execute.
- Integrate target analysis, target leakage, multicollinearity, model diagnostics, and segment analysis into one coherent target-aware workflow.
- Add reusable baseline snapshots for recurring drift and schema-change monitoring.
- Add configurable thresholds and custom checks without requiring users to fork the library.
- Improve large-dataset behavior with deterministic sampling and resource budgets.
- Add benchmark datasets and performance/accuracy regression suites.
- Stabilize the public API through the `0.1.x` series.

## Contributing

Issues and pull requests are welcome. Keep changes focused, include tests for behavioral changes, and do not commit generated reports, datasets, credentials, caches, virtual environments, or build output.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
