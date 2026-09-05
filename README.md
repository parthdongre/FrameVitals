<div align="center">

# FrameVitals

### Know if your data is healthy, stable, and ML-ready — before your model finds out.

**FrameVitals is a source-aware Python toolkit for data health, drift detection, anomaly analysis, data contracts, quality gates, and ML-readiness diagnostics on tabular data.**

[![Tests](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml/badge.svg)](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/framevitals.svg)](https://pypi.org/project/framevitals/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/parthdongre/FrameVitals?style=social)](https://github.com/parthdongre/FrameVitals)

[Installation](#installation) · [Quick Start](#quick-start) · [Workflows](#common-workflows) · [CLI](#command-line) · [Docs](docs/) · [Contributing](CONTRIBUTING.md)

</div>

---

FrameVitals helps you decide whether a dataset is **healthy enough to trust** before it reaches a model, analytics workflow, dashboard, or production pipeline.

It provides one consistent API for inspecting data quality, ML readiness, anomalies, drift, contracts, validation, snapshots, and CI-friendly quality gates.

```python
import framevitals as fv

report = fv.analyze(data)
drift = fv.compare(reference, current)
contract = fv.infer_contract(reference)
validation = fv.validate(current, contract)
gate = fv.gate(current, reference=reference, contract=contract)
```

The goal is simple: **catch bad data before it becomes a bad model, broken dashboard, or production incident.**

## Installation

Install the package from PyPI:

```bash
pip install framevitals
```

FrameVitals supports **Python 3.11, 3.12, and 3.13**.

Optional capabilities are available as extras:

```bash
pip install "framevitals[arrow]"   # Arrow and Parquet interoperability
pip install "framevitals[duckdb]"  # DuckDB relations
pip install "framevitals[plot]"    # plotting and report charts
pip install "framevitals[ml]"      # optional ML diagnostics
pip install "framevitals[ai]"      # Ollama-backed AI capabilities
pip install "framevitals[web]"     # Flask web runtime
pip install "framevitals[all]"     # all optional runtime capabilities
```

## Quick Start

Analyze a file directly:

```python
import framevitals as fv

report = fv.analyze("customers.csv")

print(report.health["overall_score"])
print(report.ml_readiness)
print(report.findings[:3])
```

Or pass a pandas DataFrame:

```python
import pandas as pd
import framevitals as fv

customers = pd.read_csv("customers.csv")
report = fv.analyze(customers)
```

FrameVitals also supports Parquet, PyArrow data, and lazy DuckDB relations when the corresponding optional dependencies are installed.

## Common Workflows

### Run only the diagnostic you need

The focused APIs let you inspect one part of a dataset without running the complete analysis pipeline:

```python
fv.profile(data)
fv.health(data)
fv.quality(data)
fv.ml_readiness(data)
fv.statistics(data)
fv.anomalies(data)
fv.relationships(data)
```

### Compare datasets for drift

```python
result = fv.compare(reference, current)

print(result.severity)
print(result["columns"][:3])
```

Use this to compare training and production data, historical batches, pipeline outputs, or any reference/current pair.

### Infer and validate a data contract

```python
contract = fv.infer_contract(reference)
result = fv.validate(current, contract)

if result.status == "fail":
    for finding in result.findings:
        print(finding["message"])
```

Contracts can capture expectations such as schema, data types, nullability, numeric bounds, allowed values, and uniqueness.

### Add a quality gate

```python
result = fv.gate(
    current,
    reference=reference,
    contract=contract,
)

print(result.status)  # pass / warn / fail
print(result.passed)
```

A gate combines the checks you choose into one verdict that can be used in scripts, pipelines, and CI.

### Add domain-specific checks

```python
@fv.check("positive revenue", severity="error")
def positive_revenue(df):
    return {
        "passed": bool((df["revenue"] >= 0).all()),
        "message": "Negative revenue values were found.",
    }

result = fv.gate(data, custom_checks=[positive_revenue])
```

Custom checks make it possible to enforce application-specific rules without modifying FrameVitals itself.

### Run target-aware analysis

```python
report = fv.analyze(
    data,
    target="churn",
    mode="deep",
)
```

Target-aware analysis can surface modelling risks such as leakage, imbalance, redundant features, multicollinearity, and weak baseline relationships.

### Create monitoring snapshots

```python
report = fv.analyze(current)
snapshot = report.snapshot("snapshot.json")
```

Compare compact snapshots later without retaining every raw dataset:

```python
previous = fv.load_snapshot("previous.json")
latest = fv.load_snapshot("snapshot.json")
change = fv.compare_snapshots(previous, latest)
```

## Analysis Modes

Choose how much work FrameVitals should perform:

```python
fv.analyze(data, mode="quick")
fv.analyze(data, mode="standard")
fv.analyze(data, mode="deep")
fv.analyze(data, mode="research")
```

Use `quick` for fast checks and the deeper modes when you want broader statistical or modelling diagnostics. For preset-driven configuration, `exhaustive` is available as an alias for the deepest built-in preset while `research` remains supported.

## Source-Aware Execution

FrameVitals is designed to work with more than pandas alone. Supported sources can include DataFrames, files, Arrow-native data, and DuckDB relations.

Where semantics allow it, large or lazy sources can use bounded or streaming execution instead of being loaded fully into pandas. Operations that require exact results can still materialize the full dataset, and execution metadata reports those decisions.

## Command Line

The Python package also includes a CLI.

Analyze a dataset:

```bash
framevitals analyze customers.csv
```

Compare two datasets:

```bash
framevitals compare reference.csv current.csv
```

Infer a contract:

```bash
framevitals infer-contract reference.csv
```

Create a monitoring snapshot:

```bash
framevitals snapshot customers.csv
```

Inspect dataset execution capabilities:

```bash
framevitals inspect customers.csv
```

See all commands and options with:

```bash
framevitals --help
```

## CI and GitHub Actions

FrameVitals can sit between your data pipeline and downstream work:

```text
Data / ETL
    ↓
FrameVitals Gate
    ↓
PASS / WARN / FAIL
    ↓
Training / Analytics / Production
```

The repository includes a reusable GitHub Action:

```yaml
- uses: parthdongre/FrameVitals@v0.3.0
  id: framevitals
  with:
    current: data/production.parquet
    reference: data/training.parquet
    contract: data/contract.json
    output: framevitals-gate.json
```

For production workflows, pin the action to a released tag or commit.

## Python API

The main workflow entry points are available directly from `framevitals`:

```python
fv.analyze(...)
fv.plan(...)
fv.profile(...)
fv.health(...)
fv.quality(...)
fv.ml_readiness(...)
fv.statistics(...)
fv.anomalies(...)
fv.relationships(...)
fv.compare(...)
fv.infer_contract(...)
fv.validate(...)
fv.check(...)
fv.run_checks(...)
fv.gate(...)
fv.create_snapshot(...)
fv.compare_snapshots(...)
```

For detailed API behaviour, configuration, source semantics, performance notes, and advanced usage, see [`docs/`](docs/).

## Development

Clone the repository and install it in development mode:

```bash
git clone https://github.com/parthdongre/FrameVitals.git
cd FrameVitals
pip install -e ".[all,dev]"
```

Run the test suite:

```bash
pytest
```

## Contributing

Contributions are welcome, including bug fixes, diagnostics, tests, documentation, integrations, and performance improvements.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Documentation

Detailed documentation lives in [`docs/`](docs/).

- [Changelog](CHANGELOG.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Issue Tracker](https://github.com/parthdongre/FrameVitals/issues)

## License

FrameVitals is released under the [MIT License](LICENSE).
