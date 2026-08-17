# Releasing FrameVitals

FrameVitals publishes to PyPI through GitHub Actions Trusted Publishing. No long-lived PyPI API token is required.

## Branch model

- `main` is release-ready; version tags and published GitHub releases come from `main`.
- `dev` is the normal integration branch.
- Stabilization branches such as `develop/august` are promoted into `dev` only after their contracts and specialized CI lanes are clean.
- A stabilization-branch merge is not itself a PyPI release.

## Release candidate checklist

Before promoting a release candidate toward `main`:

1. Core Tests, Package, minimum-dependency, platform, Arrow/fallback, native Rust, optional-feature, frontend/package-quality, docs, and security-relevant CI must be green.
2. Performance-sensitive changes must have benchmark evidence and accuracy/safety checks; do not publish unsupported universal speed claims.
3. Review generated files, temporary artifacts, stale compatibility shims, and branch-only experiments before promotion.
4. Update `CHANGELOG.md` with the release section and any known 0.x compatibility changes.
5. Keep all release-version sources synchronized:
   - GitHub release tag, e.g. `v0.2.0`;
   - `pyproject.toml` → `[project].version`;
   - `src/framevitals/__init__.py` → `__version__`;
   - `rust/framevitals-core/Cargo.toml` → `[package].version`;
   - `rust/framevitals-py/Cargo.toml` → `[package].version`.
6. Confirm the package workflow builds and validates both the portable fallback wheel and a compatible native wheel.
7. Confirm the clean-environment package smoke proves pip prefers the native wheel when both compatible native and universal fallback wheels are present.

## Distribution model

FrameVitals intentionally publishes two kinds of Python wheels:

- **Native ABI3 wheels** for supported/common platforms. These include `framevitals._native` and are preferred automatically by pip when compatible.
- **Portable fallback wheel** (`py3-none-any`) for environments without a published native wheel.

The release workflow also publishes one source distribution.

Current native release targets are:

- Linux x86_64;
- macOS arm64;
- macOS x86_64;
- Windows x86_64.

The native extension uses a Python 3.11 ABI3 floor so one compatible wheel can serve supported Python 3.11+ versions on the same platform.

## One-time PyPI setup

1. Create/sign in to the PyPI account.
2. Configure a pending Trusted Publisher for project `framevitals`.
3. GitHub owner: `parthdongre`.
4. Repository: `FrameVitals`.
5. Workflow filename: `release.yml`.
6. GitHub environment: `pypi`.
7. Create the matching `pypi` environment in GitHub repository settings; production approval is recommended.

If the repository is renamed, update the PyPI Trusted Publisher configuration before publishing again.

## Publish checklist

1. Promote the reviewed release candidate to `main` through the normal branch process.
2. Verify the exact `main` commit has the intended release versions and changelog.
3. Create a GitHub release from that exact commit with a matching tag such as `v0.2.0`.
4. Publishing the GitHub release triggers `.github/workflows/release.yml`.
5. The workflow must:
   - verify tag/Python/Rust version consistency;
   - build one fallback wheel and one source distribution;
   - build the configured native ABI3 wheels;
   - verify each native wheel actually contains `framevitals._native`;
   - run `twine check` across the complete payload;
   - publish through Trusted Publishing only after all build jobs pass.
6. Confirm the expected files appear on PyPI.
7. In a brand-new environment, run:

   ```bash
   python -m pip install --upgrade framevitals
   python -c "import framevitals; print(framevitals.__version__)"
   framevitals --version
   ```

8. On a supported native platform, also verify:

   ```bash
   python -c "from framevitals.backends import backend_status; print(backend_status())"
   ```

   `native_available` should be `True` and automatic backend selection should choose `rust`.
9. Smoke-test one normal analysis and one source-aware/Arrow path from the published wheel.

## Release discipline

Do not bump versions merely to merge a stabilization branch into `dev`. Version bumps belong to an intentional release-candidate commit.

Do not publish a release because the feature list is long. Publish when correctness, packaging, provenance, platform compatibility, and performance-sensitive behavior have all passed their release gates.
