# Stability and compatibility policy

FrameVitals follows semantic versioning, but the project is currently in the `0.x`
series. The purpose of this policy is to make that maturity level explicit without
using "alpha" as an excuse for arbitrary breakage.

## Python versions

The current package metadata supports:

- Python 3.11
- Python 3.12
- Python 3.13

The main test matrix exercises all three. Separate smoke workflows exercise public
package/CLI behavior on Windows and macOS in addition to the primary Linux CI.

## Dependency lower bounds

Minimum core dependency versions declared in `pyproject.toml` are tested in a dedicated
Python 3.11 lower-bound workflow. If FrameVitals begins relying on behavior unavailable
at an advertised lower bound, the project must either restore compatibility or raise
the declared minimum deliberately.

Optional capability groups (`arrow`, `duckdb`, `excel`, `plot`, `ml`, `ai`, `web`,
`docs`) remain independently installable. Core import/CLI behavior should not require
an optional dependency merely because the corresponding module exists in the source
tree.

## Public Python surface

The package-root workflow APIs are the primary public surface:

- `inspect_source`
- `analyze` / `plan`
- focused diagnostics such as `profile`, `health`, `statistics`, and `relationships`
- `compare`
- `infer_contract` / `validate`
- `check` / `run_checks` / `discover_checks`
- `gate`
- snapshot/history helpers

A regression test locks the exported package surface so accidental symbol removal is
caught in CI.

Modules and helpers not exported at package root may still be intentionally reusable,
but should not be assumed to have the same compatibility promise unless documented.

## Dict-compatible result objects

During `0.x`, public result objects remain subclasses of `dict` to preserve existing
mapping and JSON-oriented callers while adding helper methods.

FrameVitals avoids injecting Python-only wrapper metadata into serialized payloads.
For example, `DiagnosticResult.diagnostic` is object metadata and does not become a
new JSON key merely because a focused result is wrapped.

## Versioned machine-readable schemas

Several persisted/automation-facing structures carry explicit schema versions:

- analysis results: `result_schema_version`
- monitoring snapshots: `snapshot_schema_version`
- execution provenance: `execution_schema_version`
- data contracts: contract `version`

A schema-version bump should accompany an intentionally incompatible machine-readable
change. New additive fields do not necessarily require a bump when older consumers
can ignore them safely.

## Execution semantics

Source-aware execution is part of the public behavior, not an implementation detail.
When a result exposes execution provenance, FrameVitals should not silently relabel:

- sampled work as exact;
- a sample shape as source shape;
- full pandas materialization as streaming;
- approximate cardinality/duplicate estimates as exact values.

Performance optimizations, including the optional native Rust core, must preserve
public result semantics. Acceleration is not allowed to create a separate correctness
contract.

## Extension compatibility

Third-party check plugins use the `framevitals.checks` Python entry-point group.
Plugin discovery is explicit and never automatic because loading an entry point
executes provider code.

Custom data sources integrate through the `DatasetSource` / `StreamingDatasetSource`
protocols. Prefer standard Arrow interoperability over a library-specific dependency
when the protocol preserves the required metadata and execution semantics.

## What may change during 0.x

Before `1.0`, minor releases may still revise:

- scoring thresholds;
- finding codes or wording;
- result structure not protected by an explicit schema contract;
- source-adapter implementation details;
- experimental analysis modules;
- extension points that are explicitly marked provisional.

Breaking changes should be documented in `CHANGELOG.md` and release notes before they
reach `main`. Avoid breaking package-root workflow signatures unless the improvement
is substantial enough to justify migration cost.

## Deprecation direction

During `0.x`, FrameVitals may occasionally make direct breaking changes when maintaining
two behaviors would create more ambiguity than value. Where practical, prefer a
warning/migration period.

After `1.0`, the intended policy is stricter: public package-root APIs and versioned
schemas should use documented deprecation periods before removal, except when a change
is required to address a security or correctness issue.

## CI as the compatibility contract

The repository intentionally separates compatibility concerns into dedicated lanes:

- Python 3.11/3.12/3.13 core tests;
- declared minimum dependency tests;
- Arrow streaming tests;
- DuckDB interoperability tests;
- Polars-through-Arrow protocol tests;
- Windows/macOS public smoke tests;
- native Rust bridge checks;
- plugin-provider install/discovery tests;
- package/wheel validation;
- strict documentation build;
- performance guardrails;
- CodeQL security analysis.

A documented compatibility claim should ideally have a corresponding automated lane
before it is treated as a project promise.
