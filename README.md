# FrameVitals

**Diagnostics and ML-readiness checks for tabular data.**

FrameVitals is an open-source Python toolkit for inspecting tabular datasets before they reach a model or production pipeline. It combines data profiling, quality scoring, anomaly detection, statistical diagnostics, drift analysis, ML-readiness checks, target-aware model benchmarking, explainability, visualization, cleaning, and optional AI-assisted interpretation behind one API.

> Package name: `framevitals`  
> Current status: `0.1.0.dev0` — pre-alpha

## Why FrameVitals?

A dataset can be syntactically valid and still be risky to use. FrameVitals is designed to answer questions such as:

- What is structurally wrong with this dataset?
- Which columns look like identifiers, targets, dates, text, or useful numeric features?
- How much missingness, duplication, imbalance, leakage, or multicollinearity is present?
- Are there suspicious rows or distribution shifts?
- Is the dataset reasonably ready for machine learning?
- If a target is supplied, which baseline models perform best and why?

## Core capabilities

- Dataset loading for CSV, TSV, Excel, and JSON
- Structural profiling and semantic column-role inference
- Data-health scoring and ML-readiness scoring
- Missingness, duplicate, cardinality, and quality diagnostics
- Deep statistical analysis
- Ensemble anomaly detection
- Target leakage and multicollinearity checks
- Time-series and text-column profiling
- Dataset drift comparison
- Target-aware baseline modeling and model leaderboard
- SHAP-based explainability where supported
- Conservative dataset cleaning and chart generation
- Optional local-LLM/agent workflows
- CLI and Python API

## Install

FrameVitals currently targets Python 3.11 and 3.12.

### Development install

```bash
git clone https://github.com/parthdongre/DataLens-AI.git
cd DataLens-AI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,web]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The package metadata is already configured for PyPI. After the first public release is published, the intended installation command is:

```bash
pip install framevitals
```

## Python API

```python
import framevitals as fv

report = fv.analyze(
    "customers.csv",
    mode="standard",
)

print(report["health"]["overall_score"])
print(report["ml_readiness"])
```

With a supervised-learning target:

```python
report = fv.analyze(
    "customers.csv",
    target="churn",
    mode="deep",
)

print(report["model_leaderboard"])
print(report["explainability"])
```

### Analysis modes

| Mode | Intended use |
| --- | --- |
| `quick` | Fast structural and quality checks |
| `standard` | Default diagnostics plus deeper analytics |
| `deep` | More computational analysis and ML diagnostics |
| `research` | Broadest analysis budget |

## CLI

```bash
framevitals --version
framevitals analyze dataset.csv
framevitals analyze dataset.csv --mode quick
framevitals analyze dataset.csv --target churn --mode deep
```

The CLI prints a JSON summary containing the dataset health, ML-readiness result, detected signals, and timings.

## Optional web application

The repository also contains a Flask API, React dashboard, and Streamlit console built on top of the same analysis engine.

```bash
pip install -e ".[web]"
./run.sh
```

Typical local development endpoints are:

- Flask API: `http://127.0.0.1:5055`
- React dashboard: `http://127.0.0.1:5173`
- Streamlit console: `http://127.0.0.1:8501`

The web application is an interface around FrameVitals; `src/framevitals/` is the canonical Python package.

## Project layout

```text
.
├── src/framevitals/          # Canonical installable Python package
├── tests/                    # Automated test suite
├── frontend/                 # Optional React + TypeScript dashboard
├── modules/                  # Temporary backwards-compatibility shims
├── demo_datasets/            # Example datasets
├── app.py                    # Optional Flask application
├── streamlit_app.py          # Optional Streamlit console
├── pyproject.toml            # Package metadata and dependencies
└── .github/workflows/        # Test and package CI
```

The `modules/` namespace is retained temporarily for compatibility with older application code. New code should import from `framevitals.*` only.

## Development

Install the development environment and run the tests:

```bash
pip install -e ".[dev,web]"
pytest
```

Useful package checks:

```bash
python -m compileall src/framevitals
framevitals --version
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Public API

The intentionally small top-level API is:

```python
import framevitals as fv

fv.analyze(...)
fv.__version__
```

Lower-level modules remain importable from `framevitals.<module>` for advanced use, but the top-level API is kept small so it can evolve deliberately.

## Design principles

FrameVitals aims to be:

1. **Evidence-first** — return structured diagnostics rather than vague claims.
2. **Failure-tolerant** — expensive optional analyses should not sink an entire run.
3. **Target-aware** — supervised diagnostics only run when a target is meaningful.
4. **Composable** — usable from Python, a CLI, or another application.
5. **Inspectable** — outputs are ordinary Python dictionaries that can be logged, serialized, or rendered elsewhere.
6. **Open-source friendly** — package boundaries, tests, CI, contributor guidance, and release metadata are treated as product features.

## Roadmap

Near-term priorities include:

- Stabilize the public API for the first `0.1.x` release
- Reduce heavyweight dependencies through optional extras and lazy imports
- Improve large-dataset performance
- Expand drift and schema-change diagnostics
- Add more deterministic report/export formats
- Publish reproducible benchmark datasets and performance measurements
- Publish the package to PyPI

## Contributing

Issues and pull requests are welcome. Please keep changes focused, include tests for behavioral changes, and avoid committing generated runtime files or build artifacts.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).

## License

FrameVitals is released under the MIT License. See [LICENSE](LICENSE).
