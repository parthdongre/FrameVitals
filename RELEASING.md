# Releasing FrameVitals

FrameVitals is configured to publish to PyPI through GitHub Actions Trusted Publishing. No long-lived PyPI API token is required.

## One-time PyPI setup

Before the first release:

1. Create or sign in to your PyPI account.
2. Configure a pending Trusted Publisher for the `framevitals` project.
3. Use the GitHub owner `parthdongre`.
4. Use the repository name that contains FrameVitals at release time.
5. Set the workflow filename to `release.yml`.
6. Set the GitHub environment to `pypi`.
7. In GitHub repository settings, create a `pypi` environment and require manual approval for production publishing.

If the GitHub repository is renamed later, update the PyPI Trusted Publisher to match the new repository name before publishing again.

## Release checklist

1. Make sure the test and package workflows are green.
2. Update `CHANGELOG.md`.
3. Change the version in `pyproject.toml` and `src/framevitals/__init__.py` to the same release version.
4. Commit the release preparation changes.
5. Create a GitHub release with a matching version tag such as `v0.1.0`.
6. Publishing the GitHub release triggers `.github/workflows/release.yml`.
7. Confirm that the new version appears on PyPI and can be installed into a clean environment.

## Version consistency

Until version metadata is centralized, keep these two values identical:

- `pyproject.toml` → `[project].version`
- `src/framevitals/__init__.py` → `__version__`

A future cleanup can derive the package version from one canonical source.
