# Quality gates and custom checks

FrameVitals quality gates combine the checks a pipeline actually needs into one
`pass` / `warn` / `fail` verdict.

## Built-in gate families

A gate can run any combination of:

- exact contract validation;
- reference-vs-current drift;
- exact user-defined Python checks.

```python
result = fv.gate(
    current,
    reference=reference,
    contract=contract,
    custom_checks=[positive_revenue],
)

print(result.status)
print(result.passed)
print(result.reasons)
```

At least one family must be selected.

## Contracts

Infer a reusable contract from a trusted reference dataset:

```python
contract = fv.infer_contract(reference)
validation = fv.validate(current, contract)
```

Current contracts can encode:

- required and optional columns;
- broad dtype expectations;
- nullability and tolerated null rates;
- numeric bounds with configurable tolerance;
- low-cardinality allowed values;
- uniqueness hints.

Contract validation remains exact. If the input is Arrow-, relation-, or file-backed,
FrameVitals may materialize the complete dataset to pandas rather than weakening the
contract to a sample.

## Custom Python invariants

```python
@fv.check(
    "positive revenue",
    severity="error",
    description="Revenue cannot be negative.",
)
def positive_revenue(df):
    minimum = float(df["revenue"].min())
    return {
        "passed": minimum >= 0,
        "message": "Negative revenue found." if minimum < 0 else "Revenue is valid.",
        "details": {"minimum": minimum},
    }
```

Run checks independently:

```python
result = fv.run_checks(current, [positive_revenue])
print(result.status)
print(result.findings)
```

Or include them in a gate:

```python
result = fv.gate(current, custom_checks=[positive_revenue])
```

Every custom check receives an isolated DataFrame copy so one check cannot mutate the
input seen by later checks. Exceptions raised by provider code become structured error
results instead of aborting the entire check collection.

### Severity

Use `severity="warning"` for soft expectations and `severity="error"` for hard
invariants. A warning-only custom-check result has status `warn` but remains passing;
an error failure produces status `fail`.

## Third-party check plugins

External packages can expose reusable checks with normal Python package entry points:

```toml
[project.entry-points."framevitals.checks"]
positive_revenue = "acme_data_checks:positive_revenue"
```

FrameVitals **does not automatically import installed plugins**. Applications opt in:

```python
checks = fv.discover_checks()
result = fv.gate(current, custom_checks=checks)
```

This is an intentional trust boundary: loading an entry point executes code supplied
by the provider package.

Discovery rejects duplicate public check names and surfaces broken providers rather
than silently changing gate behavior.

## Drift thresholds

```python
result = fv.gate(
    current,
    reference=reference,
    drift_warn_on="moderate",
    drift_fail_on="severe",
)
```

Supported severities are `stable`, `minor`, `moderate`, and `severe`.

## CLI

```bash
framevitals gate production.parquet \
  --reference training.parquet \
  --contract contract.json \
  --drift-warn-on moderate \
  --drift-fail-on severe \
  --output framevitals-gate.json
```

The CLI exits `0` for pass/warn and `1` for fail.

## GitHub Action

The repository includes a reusable composite action:

```yaml
- uses: parthdongre/FrameVitals@v0.1.0
  id: framevitals
  with:
    current: production.parquet
    reference: training.parquet
    contract: contract.json
    output: framevitals-gate.json
```

For real CI pipelines, pin a released tag or commit SHA instead of a moving branch.
The action exposes `status`, `passed`, and `result-path` outputs.

## Execution transparency

The gate's top-level `execution.full_materialization` becomes true if any selected
family required full materialization. Family-specific execution blocks remain nested
under `validation`, `drift`, and `custom` so automation can distinguish why a gate
materialized or sampled data.

See [Execution provenance](execution-provenance.md) for the shared schema.
