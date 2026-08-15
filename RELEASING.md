# Releasing FrameVitals

FrameVitals is configured to publish to PyPI through GitHub Actions Trusted Publishing. No long-lived PyPI API token is required.

## Branch model

- `main` is the release-ready branch. Published releases and version tags come from `main`.
- `dev` is the integration branch for ongoing development.
- Feature and dependency-update pull requests should target `dev`.
- When a release is ready, promote the tested release changes from `dev` to `main`, update the version/changelog, and publish from `main`.

## One-time PyPI setup

Before the first release:

1. Create or sign in to your PyPI account.
2. Open your PyPI account's **Publishing** page and configure a pending Trusted Publisher for project name `framevitals`.
3. GitHub owner: `parthdongre`.
4. Repository: `FrameVitals`.
5. Workflow filename: `release.yml`.
6. GitHub environment: `pypi`.
7. In GitHub repository settings, create a `pypi` environment. Requiring manual approval for production publishing is recommended.

A pending publisher does not reserve the package name until the first successful publication. If the repository is renamed later, update the PyPI Trusted Publisher configuration before publishing again.

## Release checklist

1. Confirm the release commit is on `main` and the Tests and Package workflows are green.
2. Confirm `CHANGELOG.md` contains the release notes and date.
3. Confirm `pyproject.toml` and `src/framevitals/__init__.py` contain the same release version.
4. Confirm the built wheel imports successfully and `framevitals --version` reports the release version.
5. Create a GitHub release from `main` with a matching tag such as `v0.1.0`.
6. Publishing the GitHub release triggers `.github/workflows/release.yml`.
7. Confirm the PyPI publish workflow succeeds.
8. Install the published package in a clean environment and run `framevitals --version`.

## Version consistency

Until version metadata is centralized, keep these two values identical:

- `pyproject.toml` → `[project].version`
- `src/framevitals/__init__.py` → `__version__`

A future cleanup can derive the package version from one canonical source.
