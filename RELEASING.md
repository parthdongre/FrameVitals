# Releasing FrameVitals

FrameVitals is configured to publish to PyPI through GitHub Actions Trusted Publishing. No long-lived PyPI API token is required.

## Branch model

- `main` is the release-ready branch. Published releases and version tags come from `main`.
- `dev` is the integration branch for ongoing development.
- Feature and dependency-update pull requests should target `dev`.
- Large stabilization branches should be promoted to `dev` only after their public contracts and specialized CI lanes are clean.
- When a release is ready, promote the tested release changes from `dev` to `main`, update the version/changelog, and publish from `main`.

## Stabilization branch → `dev`

Promoting a development branch such as `develop/august` to `dev` is **not** a PyPI release and does not require changing the package version.

Before opening or merging the promotion pull request:

1. Confirm the branch is not behind `dev`, or deliberately reconcile any new `dev` commits.
2. Confirm the package-root public surface test passes and any intended API change is documented.
3. Confirm execution/result schema changes are additive or carry the appropriate schema-version change.
4. Confirm `CHANGELOG.md` records user-facing changes under `Unreleased`.
5. Confirm the core Tests and Package workflows are green.
6. Confirm the specialized compatibility lanes relevant to the branch are green:
   - documentation strict build;
   - developer/pre-commit guardrails;
   - declared minimum dependencies;
   - Windows/macOS platform smoke;
   - Arrow/DuckDB interoperability;
   - Polars-through-Arrow interoperability;
   - external check-plugin installation/discovery;
   - native Rust bridge, when touched;
   - reusable Gate Action smoke tests, when touched;
   - performance guardrails for performance-sensitive changes.
7. Confirm no known security-analysis failure is being carried into `dev`; CodeQL runs on `dev`, `main`, and their pull requests.
8. Review the branch diff for generated files, temporary artifacts, stale compatibility shims, and accidental package-boundary changes.

A large promotion pull request should summarize the public API, compatibility impact, execution-semantics changes, optional dependencies, performance implications, and remaining known limitations.

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

## `dev` → `main` release checklist

Before promoting a release candidate from `dev` to `main`:

1. Confirm the intended release candidate on `dev` has green core and specialized CI, including CodeQL.
2. Convert the relevant `Unreleased` changelog entries into a dated release section and call out any breaking 0.x change explicitly.
3. Set the release version in both `pyproject.toml` and `src/framevitals/__init__.py` and confirm they match.
4. Build distributions from the release candidate and run `python -m twine check dist/*`.
5. Inspect the wheel boundary: the `framevitals` package and `py.typed` must be present, while removed compatibility namespaces and repository-only files must not leak into the wheel.
6. Install the built wheel into a clean environment and verify at minimum:
   - `import framevitals`;
   - installed metadata version equals `framevitals.__version__`;
   - `framevitals --version` reports the release version.
7. Merge the reviewed release pull request to `main` only after its required checks are green.

## Publish checklist

1. Confirm the release commit is on `main` and the release version/changelog are final.
2. Create a GitHub release from that exact `main` commit with a matching tag such as `v0.2.0`.
3. Publishing the GitHub release triggers `.github/workflows/release.yml`.
4. The release workflow verifies that the GitHub tag, `pyproject.toml`, and `framevitals.__version__` agree before building.
5. Confirm the Trusted Publishing job succeeds and the expected files appear on PyPI.
6. Install the published package in a brand-new environment and run:

   ```bash
   python -c "import framevitals; print(framevitals.__version__)"
   framevitals --version
   ```

7. Smoke-test at least one normal analysis path from the published wheel. For a minor release that changes optional capabilities, also smoke-test the affected extra in a clean environment.

## Version consistency

Until version metadata is centralized, keep these two values identical:

- `pyproject.toml` → `[project].version`
- `src/framevitals/__init__.py` → `__version__`

Do not bump the version merely to merge a development/stabilization branch into `dev`. Bump it only as part of intentional release preparation.

A future cleanup can derive the package version from one canonical source.
