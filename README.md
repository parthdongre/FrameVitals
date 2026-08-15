# FrameVitals

**Diagnostics and ML-readiness checks for tabular data.**

FrameVitals is an open-source Python toolkit for inspecting tabular datasets before they reach a model or production pipeline. It combines structural profiling, quality scoring, statistical diagnostics, anomaly detection, drift and time-series analysis, ML-readiness checks, target-aware baseline modeling, explainability, cleaning, visualization, and optional AI-assisted interpretation behind one package.

> Package: `framevitals`  
> Status: `0.1.0.dev0` — pre-alpha

## Install

FrameVitals supports Python 3.11, 3.12, and 3.13.

For a development install from GitHub:

```bash
git clone https://github.com/parthdongre/DataLens-AI.git
cd DataLens-AI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

After the first PyPI release, the intended command is simply:

```bash
pip install framevitals
```

### Optional features

FrameVitals keeps heavyweight integrations out of the default installation.

```bash
pip install -e ".[ml]"      # XGBoost, LightGBM, PyOD, SHAP
pip install -e ".[ai]"      # Ollama client
pip install -e ".[web]"     # Flask, Streamlit, Gunicorn
pip install -e ".[all]"     # every optional runtime feature
pip install -e ".[all,dev]" # contributor/development environment
```

The core engine still works without the optional ML libraries: XGBoost, LightGBM, PyOD, and SHAP are detected lazily and their analyses fall back or skip cleanly when unavailable.

## Python API

```python
import framevitals as fv

report = fv.analyze("customers.csv", mode="standard")

print(report["health"]["overall_score"])
print(report["ml_readiness"])
```

With a supervised target:

```python
report = fv.analyze(
    "customers.csv",
    target="churn",
    mode="deep",
)

print(report["model_leaderboard"])
print(report["explainability"])
```

## CLI

```bash
framevitals --version
framevitals analyze dataset.csv
framevitals analyze dataset.csv --mode quick
framevitals analyze dataset.csv --target churn --mode deep
```

`framevitals --version` and importing the top-level package are intentionally lightweight; the analytics pipeline is loaded only when an analysis is requested.

## Analysis modes

| Mode | Intended use |
| --- | --- |
| `quick` | Fast structural, quality, and ML-readiness checks |
| `standard` | Default deeper diagnostics and visual evidence |
| `deep` | Broader statistical and target-aware analysis |
| `research` | Largest analysis budget |

## Core capabilities

- CSV, TSV, Excel, and JSON loading
- Structural profiling and semantic column-role inference
- Data-health and ML-readiness scoring
- Missingness, duplicate, cardinality, and quality diagnostics
- Deep statistical analysis
- Ensemble anomaly detection
- Target leakage and multicollinearity checks
- Time-series and text-column profiling
- Dataset drift comparison
- Target-aware model leaderboard
- Explainability with deterministic fallback when SHAP is unavailable
- Conservative cleaning and chart generation
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

```bash
pip install -e ".[all,dev]"
pytest
python -m build
python -m twine check dist/*
```

CI tests the core installation across Python 3.11–3.13, separately smoke-tests the optional feature set, validates dependency consistency, and installs the built wheel in a clean virtual environment.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Packaging and releases

The distribution is built from the `src/` layout and explicitly excludes the legacy `modules/` compatibility namespace. GitHub Actions validates the wheel on every push and pull request.

Release publishing is configured for PyPI Trusted Publishing. See [RELEASING.md](RELEASING.md) for the one-time PyPI/GitHub environment setup and release checklist.

## Roadmap

Near-term priorities:

- Stabilize the public API for `0.1.x`
- Finish migrating application imports away from compatibility shims
- Resolve remaining advisory dead-code lint findings
- Add schema-change and drift regression tests
- Improve large-dataset performance
- Publish benchmark datasets and performance measurements
- Rename the GitHub repository to match the FrameVitals project identity
- Publish the first PyPI release

## Contributing

Issues and pull requests are welcome. Keep changes focused, include tests for behavioral changes, and do not commit generated reports, datasets, credentials, caches, virtual environments, or build output.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
