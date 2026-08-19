# FrameVitals Benchmark and Acceptance Plan

A comprehensive analysis library needs explicit gates for performance and correctness. This document proposes how new modules should be evaluated before they enter standard/deep/exhaustive presets.

## Why this matters

FrameVitals should not equate “more analysis” with “better analysis.” A new method can be valuable only if it:
- produces useful additional evidence
- behaves correctly on edge cases
- has predictable resource cost
- degrades gracefully
- does not make ordinary analysis disproportionately slower

## Benchmark dimensions

Each candidate analysis should be measured across:

### Correctness
- expected findings on golden datasets
- no false structural assumptions on unsupported data
- stable JSON/schema output
- deterministic results when seeded
- correct handling of missing/infinite/mixed values

### Runtime
- stage wall time
- percentage of total analysis time
- cold vs warm/cache-aware time where relevant

### Memory
- peak resident memory where measurable
- temporary dataframe/array copies
- model/runtime memory for optional ML/DL features

### Scalability
- tall datasets
- wide datasets
- high-cardinality categorical data
- long text columns
- large numeric matrices

### Reproducibility
- package version
- config hash
- random seed
- backend/runtime
- optional pack versions
- model ID/version/checksum

## Proposed benchmark dataset families

Synthetic/golden fixtures should cover at least:

1. **Clean mixed tabular** — representative numeric + categorical + date columns.
2. **Missingness-heavy** — MCAR-like, patterned, and column-dependent missingness.
3. **Duplicate/key issues** — duplicate rows, duplicate candidate IDs, conflicting records.
4. **High cardinality** — IDs, codes, near-unique categoricals.
5. **Outlier/anomaly** — univariate and multivariate injected anomalies.
6. **Drift numeric** — mean/variance/tail/shape changes.
7. **Drift categorical** — frequency shifts, new/missing categories.
8. **Target classification** — balanced and imbalanced labels, leakage and proxy leakage.
9. **Target regression** — nonlinear and linear signal, leakage cases.
10. **Time series** — regular/irregular timestamps, trend, seasonality, gaps, anomalies.
11. **Text** — short categories, long text, duplicates, semantic shift where model packs are tested.
12. **Adversarial dtype** — numeric-looking strings, mixed dates, booleans, infinities, malformed input.
13. **Wide** — hundreds/thousands of columns with modest rows.
14. **Tall** — large row counts with a small/medium number of columns.

## Standard size classes

Use named classes rather than hardcoding expectations around one machine:

| Class | Example purpose |
|---|---|
| Tiny | unit/smoke behavior |
| Small | interactive notebook/CLI usage |
| Medium | common local analytics workload |
| Large | sampling/streaming/backend strategy validation |
| Wide | column-scaling validation |

Exact row/column counts should be defined in the benchmark suite and can evolve as performance improves.

## Per-analysis cost classes

Every analysis registry entry should declare an expected cost class:

- `tiny` — metadata/simple cached values
- `low` — one lightweight vectorized pass
- `medium` — correlations/statistical tests/model-like operations
- `high` — ensemble, large pairwise relationships, expensive transforms
- `very_high` — deep models, embeddings, exhaustive pairwise/search procedures

The planner uses this declaration together with observed dimensions and configuration budgets.

## Default-preset admission rules

### Quick
An analysis belongs in `quick` only if it is:
- highly reliable
- broadly applicable
- low cost
- useful without extensive interpretation

### Standard
An analysis belongs in `standard` when:
- value is high for ordinary datasets
- cost is bounded/predictable
- it has strong fallback behavior

### Deep
An analysis belongs in `deep` when:
- it adds meaningful evidence beyond standard
- higher cost is justified
- output is still interpretable

### Exhaustive
An analysis can be available in `exhaustive` when:
- applicability is explicit
- resource estimates exist
- optional requirements are declared
- failures can be isolated

Being present in the codebase does not automatically qualify an analysis for any preset.

## Regression policy

Performance changes should be reviewed at the stage level rather than only using total runtime.

Flag changes when:
- an existing stage becomes materially slower without an intentional tradeoff
- memory copies increase unexpectedly
- a default preset begins running a previously optional expensive module
- cache/context reuse regresses

Do not enforce overly tight microbenchmark thresholds in noisy CI environments. Prefer broad regression bands, repeated samples, and trend reporting.

## Model acceptance rules

An optional ML/DL diagnostic should have a stronger acceptance bar.

Before inclusion:
1. define exactly which analytical question it answers
2. compare against simpler deterministic/classical baselines
3. measure incremental detection/value, not only model metrics
4. measure CPU runtime and memory
5. document GPU behavior if supported
6. test deterministic seeds where possible
7. document failure/fallback behavior
8. verify model/runtime license compatibility
9. record model/version metadata in results

A deep model should not be enabled in the standard preset simply because it performs slightly better on one benchmark.

## Report-quality tests

Generated output is part of correctness.

Test:
- JSON serialization
- HTML contains required sections
- no missing/invalid links/assets for self-contained reports
- terminal renderer handles narrow/non-color terminals
- privacy mode does not expose raw sensitive examples
- report generation does not re-run expensive analysis

## Backend parity

As Polars/PyArrow/Narwhals support grows, parity tests should focus on **semantic equivalence**, not byte-for-byte output.

Examples:
- same column semantic roles
- same missing/duplicate counts
- same contract verdicts
- same drift severity class within numerical tolerance
- same findings where algorithms are equivalent

Backend-specific optimized approximations should clearly record their method.

## Suggested benchmark command surface

Future developer tooling could expose:

```bash
python -m benchmarks.run --suite core
python -m benchmarks.run --suite drift
python -m benchmarks.run --suite large
python -m benchmarks.compare baseline.json candidate.json
```

A user-facing `framevitals benchmark` command is not necessary initially; these are primarily development/release tools.

## Release gate summary

Before a significant analysis capability moves into a default preset:
- correctness fixtures pass
- serialization/report tests pass
- runtime/memory impact is understood
- fallback behavior is tested
- applicability is registered
- documentation explains the result
- no silent install/download behavior exists

This keeps “complete and exhaustive” compatible with “fast and trustworthy.”
