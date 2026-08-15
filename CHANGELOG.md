# Changelog

All notable user-facing changes to FrameVitals will be documented here.

The project is currently pre-alpha, so the public API may still change before the first stable release.

## Unreleased

### Added

- Installable `framevitals` package under `src/framevitals/`
- Public `framevitals.analyze()` API
- `framevitals` command-line interface
- Package and test GitHub Actions workflows
- Package-boundary tests preventing legacy `modules.*` imports from leaking into the distribution
- Open-source contributor, security, conduct, and issue/PR guidance

### Changed

- Project identity and package documentation moved from DataLens AI toward FrameVitals
- Reusable dashboard/report helpers moved into the canonical package
- Python dependency metadata consolidated in `pyproject.toml`

### Removed

- Academic report, presentation, and whitepaper artifacts from the development branch
- Generated TypeScript compiler metadata and generated Vite JavaScript/config declarations

## 0.1.0.dev0

Initial development version of the FrameVitals package refactor.
