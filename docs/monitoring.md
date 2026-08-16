# Monitoring with snapshots

FrameVitals snapshots turn a full analysis result into compact, versioned monitoring
state. They are designed for workflows where you want to detect meaningful changes
without storing every raw production batch.

## Create a snapshot in Python

```python
import framevitals as fv

report = fv.analyze("production.parquet", mode="quick")
snapshot = report.snapshot("production.snapshot.json")
```

A snapshot retains compact state such as:

- source identity;
- schema and dtypes;
- missingness;
- duplicate rate;
- health score;
- ML-readiness score;
- finding codes;
- analysis configuration;
- a deterministic state fingerprint.

It does **not** embed the full raw dataset.

## Snapshot from the CLI

```bash
framevitals snapshot production.parquet \
  --mode quick \
  --output production.snapshot.json
```

Snapshot generation always disables filesystem analysis artifacts. The requested
snapshot file is the only explicit monitoring artifact written by the command.

Use JSON on stdout when integrating with another tool:

```bash
framevitals snapshot production.parquet \
  --mode quick \
  --format json \
  --output production.snapshot.json
```

## Compare two snapshots

```bash
framevitals compare-snapshots \
  baseline.snapshot.json \
  production.snapshot.json
```

The comparison reports:

- whether the state fingerprint changed;
- added or removed columns;
- dtype changes;
- per-column missingness changes;
- health-score delta;
- ML-readiness delta;
- new findings;
- resolved findings.

For machine-readable output:

```bash
framevitals compare-snapshots \
  baseline.snapshot.json \
  production.snapshot.json \
  --format json \
  --output snapshot-diff.json
```

## Fail CI when state changes

Snapshot change is not automatically treated as failure because many changes are
legitimate. When a pipeline explicitly wants fingerprint stability, opt in:

```bash
framevitals compare-snapshots \
  baseline.snapshot.json \
  production.snapshot.json \
  --fail-on-change
```

The command returns exit code `1` when the fingerprints differ and `0` when they are
equal.

For data-quality enforcement, prefer `framevitals gate` when you want semantic
contract/drift thresholds rather than a strict "anything changed" rule.

## Persistent history in Python

For repeated local monitoring, `SnapshotHistory` stores compact snapshots under one
directory:

```python
history = fv.SnapshotHistory(".framevitals/history")
history.add(report.snapshot(), label="production")

latest = history.latest()
previous = history.previous()
change = history.compare_latest()
```

The default `.framevitals/` runtime directory is ignored by Git.

## Snapshot compatibility

Snapshots carry their own schema version. `fv.load_snapshot()` validates that version
before returning an `AnalysisSnapshot`, so incompatible monitoring state fails loudly
instead of being compared with unknown semantics.

During the `0.x` series, snapshot/result schemas may still evolve. Breaking changes
must be recorded in the changelog and release notes before promotion to `main`.
