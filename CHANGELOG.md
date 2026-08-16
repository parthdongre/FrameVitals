# Changelog

All notable user-facing changes to FrameVitals are documented here.

FrameVitals follows semantic versioning while the public API matures. The 0.x series may still include breaking changes, which will be called out in release notes.

## Unreleased

Development continues on the development branches. Changes intended for the next release should be documented here before they are promoted to `main`.

### Added

- Data-contract inference through `framevitals.infer_contract()` with versioned contract schemas, tolerant null-rate and numeric-bound expectations, categorical domains, optional-column semantics, and uniqueness hints
- Structured exact contract validation through `framevitals.validate()`
- `framevitals.gate()` for one CI-friendly pass/warn/fail verdict combining validation and drift
- `framevitals gate` CLI command with configurable drift warning/failure thresholds, optional contract enforcement, JSON output, and CI exit codes
- `framevitals plan` for previewing applicable modules, execution budgets, and source planning without running the heavy pipeline
- Compact versioned analysis snapshots with deterministic fingerprints, JSON persistence, and snapshot-to-snapshot schema/health/ML-readiness/finding diffs
- Dict-compatible result objects including `AnalysisResult`, `DriftResult`, `ValidationResult`, `GateResult`, and per-column result views
- Focused public APIs for profiling, column roles, health, ML readiness, quality, statistics, anomalies, relationships, and target analysis without running unrelated pipeline stages
- Source abstraction with metadata inspection, projection, and streaming capability discovery
- Bounded Arrow-backed Parquet execution for public profiling, health, ML-readiness, roles, quality, statistics, anomaly, relationship, target-aware, drift, gate, planning, and full-analysis paths where those diagnostics support bounded execution
- Optional Arrow-backed CSV and TSV streaming with exact row-count metadata, projected record batches, and pandas fallback for incompatible inputs
- Execution provenance that distinguishes exact full-stream metrics, bounded row samples, estimates, and full materialization
- Optional Rust native core and Python bridge for accelerated streaming numeric state, cardinality/quantile sketches, and backend routing
- Dedicated Arrow/NumPy-fallback and Rust/native CI coverage
- Benchmark harnesses for scale-sensitive profiling behavior
- PEP 561 `py.typed` marker for downstream type-checking support
- Optional `excel` dependency group for XLS/XLSX readers

### Changed

- Full `framevitals.analyze()` now dispatches streaming-capable sources through a bounded source-aware pipeline when artifacts do not require materialization
- Drift comparison now preserves exact source shape metadata while bounding value-distribution work for streaming sources
- `framevitals.gate()` reuses the same canonical drift and validation engines instead of maintaining separate data-loading logic
- Contract validation explicitly remains exact for uniqueness, allowed-value, nullability, and bound constraints rather than silently weakening them to sampled checks
- `framevitals.api` is now a lazy compatibility facade over the canonical focused, analysis, planning, and operations engines instead of a second eager implementation
- The installed CLI now routes directly through canonical APIs and no longer relies on a runtime monkeypatch shim
- CLI `--version` works from a no-dependencies wheel smoke install without importing the analytics stack
- The `all` extra now actually includes the Arrow capability
- Excel engines moved out of the default dependency set into `framevitals[excel]`; AI-only Pydantic moved into `framevitals[ai]`
- Excel loading now reports a FrameVitals-specific install hint when the optional reader capability is missing
- README and roadmap now describe the source-aware analyze → compare → validate → gate → monitor direction and document actual CLI exit semantics
- Package CI now requires the `py.typed` marker to be present in built wheels

### Fixed

- Streaming ML-readiness results now disclose when duplicate rate is estimated from a bounded sample instead of reporting sampled data as exact
- Streaming relationship metadata reports true source size instead of sample size
- Streaming column-role cardinality metadata distinguishes bounded-sample cardinality from full-stream native estimates
- CSV/TSV streaming sources preserve `FileSource` compatibility for existing callers and tests
- Dedicated Arrow CI now exercises streaming statistics, drift, CSV/TSV sources, and the quality-gate CLI

## 0.1.0 - 2026-08-15

First public alpha release of FrameVitals.

### Added

- Installable `framevitals` package under `src/framevitals/`
- Public `framevitals.analyze()` API for pandas DataFrames and supported dataset files
- Public `framevitals.compare()` API for reference-vs-current drift analysis
- `framevitals` command-line interface with analyze and compare commands
- Data-health and ML-readiness scoring
- Structural profiling and semantic column-role inference
- Missingness, duplicate, cardinality, statistical, anomaly, target-aware, time-series, and text diagnostics
- Drift comparison using PSI plus numeric/categorical statistical tests
- Optional artifact generation for reusable Python workflows
- Optional ML, AI, and Flask/React web dependency groups
- Python 3.11, 3.12, and 3.13 test matrix
- Package-build, wheel-boundary, Twine, and clean-install validation
- PyPI Trusted Publishing release workflow
- Dependabot configuration for Python, npm, and GitHub Actions
- Open-source contributor, security, conduct, issue, and pull-request guidance

### Changed

- Project identity and package documentation moved from DataLens AI to FrameVitals
- Reusable dashboard/report helpers moved into the canonical package
- Top-level package import and CLI version path load the analytics pipeline lazily
- Heavy ML, Ollama, and web dependencies moved into optional extras
- Reusable Python analysis no longer writes cleaned datasets or charts unless `artifacts=True`
- Repository layout is package-first, with the Flask API and React dashboard kept as optional interfaces
- Frontend module documentation now points directly at canonical `src/framevitals/` implementations
- Development setup standardized on `.venv`

### Removed

- Unused runtime dependencies including Optuna, imbalanced-learn, Pingouin, Plotly, Missingno, fpdf2, Jinja2, Joblib, and Loguru
- Academic report, presentation, and whitepaper artifacts
- Generated TypeScript compiler metadata and generated Vite JavaScript/config declarations
- Tracked runtime-output directories for uploads, reports, and cleaned datasets
- Large legacy demo CSVs and their one-off inspection harness
- Redundant Streamlit console and its configuration
- Legacy shell launcher/install scripts and duplicate `requirements.txt` development wrapper
- Deprecated top-level `modules/` compatibility namespace after application imports moved to `framevitals.*`
