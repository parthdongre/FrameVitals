# Contributing to FrameVitals

Thanks for considering a contribution to FrameVitals.

FrameVitals is being built as a package-first data-health and quality-gate library. Contributions are most valuable when they make the public workflow more reliable, source-aware, transparent, and maintainable rather than simply adding more unrelated diagnostics.

## Development setup

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

Run the test suite before making changes:

```bash
pytest
```

## Branch workflow

`main` is kept release-ready. Ongoing development is integrated through `dev`.

1. Start from the latest `dev` branch.
2. Create a focused feature or fix branch.
3. Add or update tests for behavioral changes.
4. Open the pull request against `dev`.
5. Promote tested release changes from `dev` to `main` through a release pull request.

## Architecture boundaries

`src/framevitals/` is the canonical Python package. New reusable code belongs there and should import through the `framevitals.*` namespace.

The main layers have distinct responsibilities:

| Layer | Responsibility |
| --- | --- |
| `sources.py` | Normalize inputs and expose metadata, projection, loading, and optional streaming |
| `focused.py` | Run one requested diagnostic without invoking unrelated pipeline stages |
| `analysis_api.py` | Dispatch full analysis through the appropriate source-aware execution path |
| `streaming_pipeline.py` | Orchestrate bounded full analysis for streaming-capable sources |
| `planning_api.py` | Preview execution decisions without running heavy analyses |
| `operations.py` | Cleaning, drift, contracts, validation, and quality gates |
| `result.py` / `quality_results.py` | Dict-compatible public result objects and helpers |
| `cli.py` | Thin command-line interface over canonical package APIs |
| `app.py` / `frontend/` | Optional interfaces; they must not duplicate package logic |

A compatibility module may delegate to these layers, but it should not contain a second implementation of the same public operation.

## Source-aware execution

Do not call `source.load()` automatically just because an analysis accepts a file path.

When a `StreamingDatasetSource` can safely support the operation:

1. inspect source metadata first;
2. derive execution budgets from the **true** source shape;
3. project only required columns where practical;
4. stream or sample deterministically within the budget;
5. expose execution provenance in the returned payload.

Approximate work must be labelled honestly. If a metric comes from a bounded row sample, do not present it as exact. Conversely, checks such as uniqueness, allowed-value validation, or hard numeric bounds must not silently become sampled checks when exactness is part of the public contract.

## Optional dependencies

The base package should not import optional stacks unless their feature is actually requested.

Current capability groups include:

- `framevitals[arrow]` for Arrow-backed streaming;
- `framevitals[excel]` for XLS/XLSX readers;
- `framevitals[plot]` for chart/PDF plotting support;
- `framevitals[ml]` for heavier model/explainability integrations;
- `framevitals[ai]` for agentic AI integrations;
- `framevitals[web]` for the Flask runtime;
- `framevitals[all]` for all optional runtime capabilities.

If a new optional dependency is introduced, prefer a lazy import plus a clear install hint. Add a dependency-boundary test when accidental eager imports would make the base install heavier or break another extra.

## Tests

Useful checks include:

```bash
pytest
pytest tests/test_public_api.py
pytest tests/test_framevitals_pipeline.py
pytest tests/test_optional_dependency_boundaries.py
pytest tests/test_package_boundary.py
python -m compileall src/framevitals app.py
python -m build
python -m twine check dist/*
framevitals --version
```

For source-aware changes, also run the relevant focused suites. Examples:

```bash
pytest tests/test_parquet_streaming.py
pytest tests/test_csv_streaming.py
pytest tests/test_streaming_drift.py
pytest tests/test_statistics_streaming.py
```

For the optional React dashboard:

```bash
cd frontend
npm ci
npm run build
```

Performance-sensitive changes should use the benchmark harness rather than intuition alone:

```bash
python benchmarks/benchmark_profile_scale.py --rows 50000 --scenarios numpy auto parquet
```

## Style

- Prefer clear Python over clever Python.
- Keep the public API deliberate and small.
- Avoid hidden global state in reusable library code.
- Return structured, JSON-friendly values where practical.
- Preserve dict compatibility for public result objects during the 0.x series unless a breaking change is explicitly planned.
- Optional analyses should fail gracefully when a dependency is unavailable.
- Keep source shape, sample shape, and materialization state distinct in execution metadata.
- Use Rust only for measured hot paths; keep orchestration and public semantics readable in Python.
- Do not commit generated reports, uploads, cleaned datasets, caches, virtual environments, build output, or frontend compiler artifacts.

## Pull requests

A good pull request explains:

- the problem being solved;
- why the chosen approach is appropriate;
- what behavior changed;
- what tests were run;
- any compatibility or performance implications;
- whether the change affects exactness, sampling, optional dependencies, or public result schemas.

For large features or public-API changes, open an issue first so the design can be discussed before implementation.

By contributing, you agree that your contribution will be licensed under the project's MIT License.
