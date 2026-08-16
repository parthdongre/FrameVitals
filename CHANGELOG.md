# Changelog

All notable user-facing changes to FrameVitals are documented here.

FrameVitals follows semantic versioning while the public API matures. The 0.x series may still include breaking changes, which will be called out in release notes.

## Unreleased

Development continues on the development branches. Changes intended for the next release should be documented here before they are promoted to `main`.

### Added

- Data-contract inference through `framevitals.infer_contract()` with versioned contract schemas, tolerant null-rate and numeric-bound expectations, categorical domains, optional-column semantics, and uniqueness hints
- Structured exact contract validation through `framevitals.validate()`
- `framevitals.gate()` for one CI-friendly pass/warn/fail verdict combining contract validation, drift, and optional custom checks
- `framevitals gate` CLI command with configurable drift warning/failure thresholds, optional contract enforcement, JSON output, and CI exit codes
- Reusable root `action.yml` GitHub Action for running FrameVitals quality gates with status/result outputs
- Path-scoped smoke CI for the reusable Gate Action
- `framevitals plan` for previewing applicable modules, execution budgets, source capabilities, and bounded planning samples
- `framevitals.inspect_source()` for inspecting source shape, storage kind, format, materialization state, projection support, and streaming capability without running diagnostics
- `framevitals inspect` CLI command with terminal/JSON output and optional persisted source metadata
- Compact versioned analysis snapshots with deterministic fingerprints, JSON persistence, and snapshot-to-snapshot schema/health/ML-readiness/finding diffs
- `SnapshotHistory` for persistent compact monitoring timelines, latest/previous lookup, labels, and recent-state comparison without storing raw datasets
- Dict-compatible result objects including `AnalysisResult`, `DriftResult`, `ValidationResult`, `CheckResult`, `GateResult`, and per-column result views
- `framevitals.check()` and `framevitals.run_checks()` for exact user-defined DataFrame invariants with warning/error severities and structured findings
- Opt-in `framevitals.discover_checks()` plugin discovery through the `framevitals.checks` Python entry-point group
- Focused public APIs for profiling, column roles, health, ML readiness, quality, statistics, anomalies, relationships, and target analysis without running unrelated pipeline stages
- Source abstraction with metadata inspection, projection, and streaming capability discovery
- Bounded Arrow-backed Parquet execution for public profiling, health, ML-readiness, roles, quality, statistics, anomaly, relationship, target-aware, drift, gate, planning, and full-analysis paths where those diagnostics support bounded execution
- Optional Arrow-backed CSV and TSV streaming with exact row-count metadata, projected record batches, and pandas fallback for incompatible inputs
- Native PyArrow `Table` and `RecordBatch` inputs through the same source-aware streaming engine
- Arrow C Stream / PyCapsule-compatible table input support without requiring a library-specific adapter for every producer
- Optional lazy DuckDB relation source with exact count/schema metadata, projection pushdown, Arrow record batches, and full pandas materialization only for exact/full-row APIs
- Optional `duckdb` dependency group containing DuckDB plus its Arrow transport
- Dedicated Arrow/DuckDB interoperability CI coverage
- Versioned execution-provenance schema v1 with shared `method`, `full_materialization`, source, sampling, strategy, component, and reason fields while preserving legacy operation-specific metadata during the `0.x` migration
- Optional Rust native core and Python bridge for accelerated streaming numeric state, cardinality/quantile sketches, and backend routing
- Dedicated Arrow/NumPy-fallback and Rust/native CI coverage
- Reproducible performance benchmark workflow with time/RSS summaries and retained JSON artifacts
- PEP 561 `py.typed` marker for downstream type-checking support
- Optional `excel` dependency group for XLS/XLSX readers
- Optional `plot` dependency group for Matplotlib/Seaborn chart and report plotting support
- Optional `docs` dependency group plus a MkDocs Material documentation site covering source-aware execution, execution provenance, quality gates, custom checks, and plugin trust boundaries
- Strict path-scoped documentation CI using `mkdocs build --strict`
- Contributor architecture guidance covering canonical layers, streaming provenance, optional-dependency boundaries, and performance-sensitive changes
- Focused examples for domain-specific quality gates and persistent snapshot monitoring

### Changed

- Full `framevitals.analyze()` now resolves generic `DatasetSource` inputs and dispatches streaming-capable sources through a bounded source-aware pipeline when artifacts do not require materialization
- Drift comparison now preserves exact source shape metadata while bounding value-distribution work for streaming sources
- Flask `/api/compare` now routes through the same canonical source-aware drift engine as Python and CLI callers
- `framevitals.gate()` reuses the same canonical drift, validation, and custom-check engines instead of maintaining separate data-loading logic
- Contract validation explicitly remains exact for uniqueness, allowed-value, nullability, and bound constraints rather than silently weakening them to sampled checks
- Arbitrary custom Python checks explicitly remain exact and materialize non-pandas sources rather than silently sampling user-defined invariants
- Top-level gate execution now reports whether any selected check family required full materialization
- Materialization provenance is now based on actual execution semantics rather than assuming only file-backed sources can materialize
- Focused statistics, anomaly, relationship, role, health, ML-readiness, quality, drift, validation, custom-check, and gate paths are converging on execution-provenance schema v1 without removing legacy metadata keys
- `framevitals.api` is now a lazy compatibility facade over the canonical focused, analysis, planning, operations, check, plugin, and source-inspection engines instead of a second eager implementation
- The installed CLI now routes directly through canonical APIs and no longer relies on a runtime monkeypatch shim
- CLI `--version` works from a no-dependencies wheel smoke install without importing the analytics stack
- The `all` extra now includes Arrow and DuckDB interoperability capabilities
- Excel engines moved out of the default dependency set into `framevitals[excel]`; AI-only Pydantic moved into `framevitals[ai]`
- Matplotlib and Seaborn moved out of the default dependency set into `framevitals[plot]`
- Explainability no longer requires Matplotlib at module import time; structured importance can run without plotting and SHAP chart rendering is optional
- Flask startup no longer eagerly imports the agentic AI or report/plot stacks; those capabilities load only when their endpoints are used
- Excel loading now reports a FrameVitals-specific install hint when the optional reader capability is missing
- README and roadmap now describe the source-aware analyze → compare → validate → gate → monitor direction, source interoperability, exact-vs-bounded execution, optional capability groups, plugins, and the reusable Gate Action
- Project metadata now advertises the package as typed and points documentation users at the maintained `docs/` tree
- Package CI now requires the `py.typed` marker to be present in built wheels
- Test and package workflows now use read-only default permissions and cancel stale runs on the same branch
- Release publishing now verifies that the GitHub release tag, `pyproject.toml` version, and `framevitals.__version__` match before building/publishing

### Fixed

- CSV/TSV no-Arrow fallback no longer relies on zero-argument `super()` in a slotted dataclass subclass, restoring Python 3.11/3.12 compatibility
- Streaming ML-readiness results now disclose when duplicate rate is estimated from a bounded sample instead of reporting sampled data as exact
- Streaming relationship metadata reports true source size instead of sample size
- Streaming column-role cardinality metadata distinguishes bounded-sample cardinality from full-stream native estimates
- CSV/TSV streaming sources preserve `FileSource` compatibility for existing callers and tests
- Dedicated Arrow CI now exercises streaming statistics, drift, CSV/TSV sources, in-memory Arrow inputs, and the quality-gate CLI
- Web runtime dependency boundaries no longer force AI/report/plot packages to import during Flask startup
- Reusable Gate Action setup no longer relies on an invalid absolute `setup-python` cache dependency path
- Exact validation/custom-check provenance now reports full materialization for Arrow and relation-backed inputs when they are converted to complete pandas DataFrames

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
