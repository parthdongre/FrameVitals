# FrameVitals check plugin example

This directory is a minimal installable third-party package that contributes custom
checks to FrameVitals through standard Python entry points.

It is intentionally separate from the main `framevitals` package so the example tests
the same packaging/discovery boundary that an external project would use.

## Install beside a FrameVitals checkout

From the repository root:

```bash
python -m pip install -e .
python -m pip install -e examples/check_plugin
```

Then discover the provider checks explicitly:

```python
import framevitals as fv

checks = fv.discover_checks()
for check in checks:
    print(check.name, check.severity)
```

Use them directly or inside the quality gate:

```python
result = fv.gate(dataframe, custom_checks=checks)
print(result.summary_text())
```

## Entry-point contract

The example declares:

```toml
[project.entry-points."framevitals.checks"]
positive_revenue = "framevitals_example_checks:positive_revenue"
preferred_plan = "framevitals_example_checks:preferred_plan"
```

Provider packages may export `DataCheck` objects or compatible DataFrame callables.
FrameVitals does **not** automatically import installed providers; applications opt in
with `fv.discover_checks()` because loading entry points executes provider code.

See `docs/extending-framevitals.md` for the extension-author guide.
