# Result objects

FrameVitals public workflows return **dict-compatible result objects** during the
`0.x` series. This keeps existing mapping/JSON code working while adding discoverable
helpers for notebooks, applications, reports, and CI.

## Why dict-compatible objects?

A FrameVitals result should be easy to:

- index like a normal dictionary;
- serialize to JSON;
- pass into existing application code;
- inspect interactively with named properties and helper methods.

The `0.x` series therefore favors additive wrappers over a hard break to dataclasses
or validation models.

## Full analysis

`fv.analyze(...)` returns `AnalysisResult`.

```python
report = fv.analyze(data)

report.health
report.ml_readiness
report.findings
report.recommendations
report.column("age")
report.summary()
report.summary_text()
report.to_json("analysis.json")
report.to_html("analysis.html")
report.snapshot("snapshot.json")
```

`AnalysisResult` remains a `dict`, so existing code still works:

```python
score = report["health"]["overall_score"]
```

## Focused diagnostics

Focused APIs return `DiagnosticResult`:

- `fv.profile(...)`
- `fv.roles(...)`
- `fv.health(...)`
- `fv.ml_readiness(...)`
- `fv.quality(...)`
- `fv.statistics(...)`
- `fv.anomalies(...)`
- `fv.relationships(...)`
- `fv.target_analysis(...)`

```python
stats = fv.statistics(data, mode="quick")

stats.diagnostic       # "statistics"
stats.dataset_name
stats.execution
stats.source
stats.available
stats.summary()
stats.summary_text()
stats.to_json("statistics.json")
```

The diagnostic label is Python-side metadata and is **not injected into the mapping**.
That means wrapping a focused payload does not mutate its JSON schema:

```python
assert "diagnostic" not in stats
assert dict(stats) == stats.to_dict()
```

`to_dict()` returns a detached deep copy so callers can modify it without mutating the
live result object.

## Quality operations

### DriftResult

`fv.compare(...)` returns `DriftResult` with conveniences such as:

```python
drift.severity
drift.status
drift.columns
drift.summary_text()
drift.to_json()
```

### ValidationResult

`fv.validate(...)` returns `ValidationResult`:

```python
validation.valid
validation.status
validation.findings
validation.summary_text()
```

### CheckResult

`fv.run_checks(...)` returns `CheckResult`:

```python
checks.status
checks.passed
checks.results
checks.findings
checks.summary_text()
```

### GateResult

`fv.gate(...)` returns `GateResult`:

```python
gate.status
gate.passed
gate.checks_run
gate.reasons
gate.summary_text()
```

All four quality result types remain mapping-compatible and expose `to_dict()` /
`to_json()` through the shared quality-result base.

## Monitoring state

`AnalysisSnapshot` is a compact dict-compatible monitoring record derived from an
`AnalysisResult`. `SnapshotHistory` manages persisted snapshots without storing raw
input datasets.

```python
history = fv.SnapshotHistory(".framevitals/history")
history.add(report.snapshot(), label="production")
change = history.compare_latest()
```

## Execution metadata

Focused and quality results increasingly expose the shared execution-provenance
contract under `result.execution` or `result["execution"]`.

```python
execution = stats.execution
print(execution["execution_schema_version"])
print(execution["method"])
print(execution["sampled"])
print(execution["full_materialization"])
```

See [Execution provenance](execution-provenance.md) for the schema and exactness rules.

## Compatibility policy during 0.x

FrameVitals aims to evolve result ergonomics without needless churn:

1. result objects remain subclasses of `dict`;
2. helper properties should not silently change serialized payloads;
3. new shared metadata is added deliberately and versioned where appropriate;
4. legacy operation-specific fields remain available during migration;
5. breaking result-schema changes must be documented before release.
