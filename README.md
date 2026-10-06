<div align="center">

# FrameVitals

### Diagnose structured data, graphs, tensors, and ML models through one protocol system.

**FrameVitals is an open-source diagnostic layer for structured data and ML systems: tabular data, graphs, tensors, and PyTorch models can enter the same Prism workflow while specialized engines handle their structure safely.**

🌐 **Website:** https://framevitals.vercel.app/

[![Tests](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml/badge.svg)](https://github.com/parthdongre/FrameVitals/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/framevitals.svg)](https://pypi.org/project/framevitals/)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/parthdongre/FrameVitals?style=social)](https://github.com/parthdongre/FrameVitals)

[Website](https://framevitals.vercel.app/) · [Installation](#installation) · [Quick Start](#quick-start) · [Workflows](#common-workflows) · [CLI](#command-line) · [Docs](docs/) · [Contributing](CONTRIBUTING.md)

</div>

---

FrameVitals helps you decide whether a dataset is **healthy enough to trust** before it reaches a model, analytics workflow, dashboard, or production pipeline.

The public experience is organized around a small protocol system. **Prism** is the primary workflow: give it the data you have, plus optional context, and FrameVitals coordinates the analysis for you.

```python
import framevitals as fv

result = fv.prism(
    current,
    reference=training,
    focus="churn",
)
```

The goal is simple: **choose the intent, not the implementation.** FrameVitals coordinates the underlying analysis while keeping the full technical composition inspectable in the documentation and source code.

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
pip install "framevitals[ml]"      # optional predictive diagnostics
pip install "framevitals[graph]"   # NetworkX graph diagnostics
pip install "framevitals[onnx]"    # ONNX model graph diagnostics
pip install "framevitals[ai]"      # Ollama-backed AI capabilities
pip install "framevitals[web]"     # Flask web runtime
pip install "framevitals[all]"     # all optional runtime capabilities
```

## Quick Start

Run the **Prism Protocol** on a file:

```python
import framevitals as fv

result = fv.prism("customers.csv")

print(result.analysis.health["overall_score"])
print(result.beacons[:3])
```

Give Prism more context when you have it:

```python
result = fv.prism(
    "production.parquet",
    reference="training.parquet",
    target="churn",
)
```

Prism remains one call whether it is working from a single dataset or coordinating a richer reference-aware workflow.

FrameVitals also supports Parquet, PyArrow data, and lazy DuckDB relations when the corresponding optional dependencies are installed.

Prism can also recognize non-tabular structured objects:

```python
import numpy as np
import framevitals as fv

tensor_report = fv.prism(np.random.randn(256, 64))

# With NetworkX installed:
graph_report = fv.prism(graph)
graph_file_report = fv.prism("network.graphml")

# If PyTorch is already installed in your environment:
model_report = fv.prism(model)

# Observe one bounded forward/backward pass without taking over training:
runtime_report = fv.prism(
    model,
    sample_batch=x,
    targets=y,
    loss_fn=criterion,
    backward=True,
)

# Metadata-first checkpoint inspection without loading tensor bodies:
checkpoint_report = fv.prism("model.safetensors")

# ONNX graph + initializer diagnostics:
onnx_report = fv.prism("model.onnx")

# Nested structured data:
nested_report = fv.prism({"users": [{"id": 1}, {"id": 2}]})

# Multi-table projects:
project_report = fv.prism({
    "customers": customers,
    "orders": orders,
})
```

The graph engine uses bounded connectivity, PageRank, approximate betweenness,
community, cut-structure, clustering, k-core, spectral-connectivity, and sampled
shortest-path diagnostics. Transformer and recurrent PyTorch models receive
architecture-aware checks in addition to generic parameter/runtime analysis.
Weighted graphs with a reliable non-negative `weight` attribute automatically
use Dijkstra for the sampled path analysis. PyTorch remains optional; FrameVitals
does not install a deep-learning runtime merely to inspect a model.

## Protocols

The protocol surface is intentionally small:

```python
fv.prism(data, reference=reference, focus="churn")
fv.axiom(reference, current=current)
fv.forge(data, apply=False)
fv.tide(reference, current)
fv.pulse(data, destination="pulse.json")
```

- **Prism** — the primary comprehensive workflow.
- **Axiom** — establish and test dataset expectations.
- **Forge** — prepare and optionally apply conservative transformations.
- **Tide** — read change between two dataset states.
- **Pulse** — capture compact health states over time.

The lower-level functions remain available for advanced and compatibility use. Their composition, execution semantics, and protocol mapping are documented in [Protocol architecture](docs/protocols.md).

## Advanced Workflows

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

FrameVitals is designed to work with more than pandas alone. Mature tabular sources include DataFrames, files, Arrow-native data, and DuckDB relations; the structured adapter layer now also recognizes NetworkX graphs, NumPy/PyTorch tensors, and PyTorch models.

Where semantics allow it, large or lazy sources can use bounded or streaming execution instead of being loaded fully into pandas. Operations that require exact results can still materialize the full dataset, and execution metadata reports those decisions.

## Command Line

The CLI follows the same protocol surface:

```bash
framevitals prism customers.csv
framevitals prism production.parquet --reference training.parquet --focus churn
framevitals axiom training.parquet --current production.parquet
framevitals forge customers.csv
framevitals tide training.parquet production.parquet
framevitals pulse production.parquet --output pulse.json
```

The terminal keeps protocol execution intentionally high-level. Detailed method composition belongs in the documentation, not in routine command output.

The legacy low-level commands remain available during the compatibility window.

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

For most users, start with the protocol surface:

```python
fv.prism(...)
fv.axiom(...)
fv.forge(...)
fv.tide(...)
fv.pulse(...)
```

Focused and lower-level APIs remain available when you need direct control over a specific operation.

For protocol composition, detailed API behaviour, configuration, source semantics, performance notes, and advanced usage, see [`docs/`](docs/).

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
