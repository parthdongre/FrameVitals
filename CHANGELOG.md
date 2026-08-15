# Changelog

All notable user-facing changes to FrameVitals will be documented here.

The project is currently pre-alpha, so the public API may still change before the first stable release.

## Unreleased

### Added

- Installable `framevitals` package under `src/framevitals/`
- Public `framevitals.analyze()` API
- `framevitals` command-line interface
- Core test matrix for Python 3.11, 3.12, and 3.13
- Package-build and distribution-boundary validation
- PyPI Trusted Publishing release workflow
- Dependabot configuration for Python, npm, and GitHub Actions
- Open-source contributor, security, conduct, issue, and pull-request guidance

### Changed

- Project identity and package documentation moved from DataLens AI toward FrameVitals
- Reusable dashboard/report helpers moved into the canonical package
- Top-level package import and CLI version path now load the analytics pipeline lazily
- Heavy ML, Ollama, and web dependencies moved into optional extras
- Development setup standardized on `.venv`
- Legacy `modules/` implementations converted to compatibility shims where migrated

### Removed

- Unused runtime dependencies including Optuna, imbalanced-learn, Pingouin, Plotly, Missingno, fpdf2, Jinja2, Joblib, and Loguru
- Academic report, presentation, and whitepaper artifacts from the development branch
- Generated TypeScript compiler metadata and generated Vite JavaScript/config declarations

## 0.1.0.dev0

Initial development version of the FrameVitals package refactor.
