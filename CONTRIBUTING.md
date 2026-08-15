# Contributing to FrameVitals

Thanks for considering a contribution to FrameVitals.

## Development setup

```bash
git clone https://github.com/parthdongre/FrameVitals.git
cd FrameVitals
git switch dev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Run the test suite before making changes:

```bash
pytest
```

## Branch workflow

`main` is kept release-ready. Ongoing development is integrated through `dev`.

1. Start from the latest `dev` branch.
2. Create a focused feature or fix branch.
3. Add or update tests for behavioral changes.
4. Open the pull request against `dev`.
5. Promote tested release changes from `dev` to `main` through a release pull request.

## Source layout

`src/framevitals/` is the canonical Python package. New reusable code belongs there and should import through the `framevitals.*` namespace.

The Flask API and React dashboard are optional interfaces around the package. Product logic should stay in `src/framevitals/` rather than being duplicated in application code.

## Tests

Useful checks include:

```bash
pytest
pytest tests/test_public_api.py
pytest tests/test_framevitals_pipeline.py
pytest tests/test_package_boundary.py
python -m compileall src/framevitals app.py
python -m build
python -m twine check dist/*
framevitals --version
```

For the optional React dashboard:

```bash
cd frontend
npm ci
npm run build
```

## Style

- Prefer clear Python over clever Python.
- Keep the public API deliberate and small.
- Avoid hidden global state in reusable library code.
- Return structured, JSON-friendly values where practical.
- Optional analyses should fail gracefully when a dependency is unavailable.
- Do not commit generated reports, uploads, cleaned datasets, caches, virtual environments, build output, or frontend compiler artifacts.

## Pull requests

A good pull request explains:

- the problem being solved;
- why the chosen approach is appropriate;
- what behavior changed;
- what tests were run;
- any compatibility or performance implications.

For large features or public-API changes, open an issue first so the design can be discussed before implementation.

By contributing, you agree that your contribution will be licensed under the project's MIT License.
