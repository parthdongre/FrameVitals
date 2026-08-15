# Changelog

All notable user-facing changes to FrameVitals are documented here.

FrameVitals follows semantic versioning while the public API matures. The 0.x series may still include breaking changes, which will be called out in release notes.

## Unreleased

Development continues on the `dev` branch.

### Added

- Development work on **FrameVitals Contracts** for inferred data-health expectations and reusable dataset validation.
- `framevitals.infer_contract()` and `framevitals.validate()` development APIs.
- CI-friendly contract inference/validation CLI work with structured pass/warn/fail evidence.

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
- Optional ML, AI, and web dependency groups
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
- Development setup standardized on `.venv`
- Legacy `modules/` implementations converted to compatibility shims where migrated

### Removed

- Unused runtime dependencies including Optuna, imbalanced-learn, Pingouin, Plotly, Missingno, fpdf2, Jinja2, Joblib, and Loguru
- Academic report, presentation, and whitepaper artifacts
- Generated TypeScript compiler metadata and generated Vite JavaScript/config declarations
