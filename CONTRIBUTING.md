# Contributing to FrameVitals

Thanks for considering a contribution to FrameVitals.

## Development setup

```bash
git clone https://github.com/parthdongre/DataLens-AI.git
cd DataLens-AI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,web]"
```

Run the test suite before making changes:

```bash
pytest
```

## Source layout

`src/framevitals/` is the canonical Python package. New library code should live there and should import other library code through the `framevitals.*` namespace.

`modules/` exists only as a temporary compatibility layer for older application code. Do not add new functionality there.

## Making a change

1. Create a focused branch from the current development base.
2. Keep the change small enough to review.
3. Add or update tests for behavioral changes.
4. Run the relevant tests locally.
5. Update documentation when public behavior changes.
6. Open a pull request describing the problem, the solution, and how it was verified.

For large features or public-API changes, open an issue first so the design can be discussed before implementation.

## Tests

The full suite is:

```bash
pytest
```

Useful focused checks include:

```bash
pytest tests/test_public_api.py
pytest tests/test_framevitals_pipeline.py
pytest tests/test_package_boundary.py
python -m compileall src/framevitals
framevitals --version
```

## Style

- Prefer clear Python over clever Python.
- Keep the public API deliberate and small.
- Avoid hidden global state in reusable library code.
- Return structured, JSON-friendly values where practical.
- Do not make optional analyses crash the entire pipeline when they can fail gracefully.
- Do not commit generated reports, uploads, cleaned datasets, caches, virtual environments, build output, or frontend compiler artifacts.

## Pull requests

A good pull request explains:

- What problem is being solved?
- Why is the chosen approach appropriate?
- What behavior changed?
- What tests were run?
- Are there compatibility or performance implications?

By contributing, you agree that your contribution will be licensed under the project's MIT License.
