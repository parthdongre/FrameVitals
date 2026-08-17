# Changelog

All notable user-facing changes to FrameVitals are documented here.

FrameVitals follows semantic versioning while the public API matures. The 0.x series may still include breaking changes, which are called out in release notes.

## Unreleased

No user-facing changes are currently queued beyond 0.2.0.

## 0.2.0 - 2026-08-17

FrameVitals 0.2.0 is the first release built around the source-aware Arrow/Rust execution architecture rather than treating every dataset as a pandas-first workload.

### Highlights

- Added bounded Arrow streaming for large Parquet, CSV/TSV, PyArrow, Arrow C Stream, Polars-through-Arrow, and optional DuckDB relation inputs.
- Added a native Rust execution backend with direct Arrow `RecordBatch` profiling, mergeable numeric state, native categorical sketches, and full-stream log-quantile sketches.
- Added exact-once reuse so downstream Deep/Research diagnostics consume already-known full-stream facts instead of recomputing weaker bounded-sample estimates.
- Added exact full-stream count, missingness, mean, variance/std, min/max, skewness, and excess kurtosis through mergeable central moments up to M4.
- Replaced fixed evenly spaced row sampling with deterministic stratified-jitter sampling to reduce periodic/structured-data aliasing while preserving reproducibility and row order.
- Added explicit execution provenance for full-stream, projected-column, sketch, bounded-sample, and materialized operations.
- Added physical large-scale benchmark coverage up to 500,000 × 10,000 (5 billion logical cells) with native/fallback routing, accuracy, exact-once, and no-full-materialization checks.
- Added mixed statistical ground-truth validation across native and fallback execution.
- PyPI publishing now builds native ABI3 wheels for common Linux/macOS/Windows targets while retaining a portable pure-Python fallback wheel and source distribution.

### Analysis and public APIs

- Added focused APIs for `profile`, `roles`, `health`, `ml_readiness`, `quality`, `statistics`, `anomalies`, `relationships`, and `target_analysis` so callers do not need to run unrelated pipeline stages.
- Added `framevitals.plan()` for previewing execution budgets, source capabilities, applicable work, and bounded planning behavior.
- Added `framevitals.inspect_source()` and the matching CLI command for source metadata/capability inspection without analysis.
- Added data-contract inference with `framevitals.infer_contract()` and structured validation with `framevitals.validate()`.
- Added `framevitals.gate()` plus a reusable GitHub Action for CI-friendly contract, drift, and custom-check verdicts.
- Added `framevitals.check()`, `run_checks()`, and opt-in third-party check discovery through Python entry points.
- Added compact versioned analysis snapshots, snapshot history, and snapshot-to-snapshot monitoring diffs.
- Added structured dict-compatible result objects including analysis, diagnostic, drift, validation, check, and gate results.
- Added `system-info`, snapshot, monitoring, planning, inspection, validation, and gate CLI workflows.

### Execution architecture

- Full `framevitals.analyze()` now resolves generic dataset sources and uses bounded source-aware execution when the source supports streaming/projection.
- Ultra-wide sources use deterministic schema projection under explicit cell/column budgets instead of scanning every source cell blindly.
- Streaming analysis performs one authoritative full-source profile scan and schedules only genuinely row-dependent modules on the bounded working sample.
- Numeric native execution accepts Arrow record batches directly through the Arrow C Data/PyCapsule interface, avoiding the former Arrow → NumPy float64 → per-column Python bridge on supported types.
- Streaming numeric profiling uses a specialized moments + log-quantile state instead of paying for unrelated HLL/heavy-hitter/reservoir work on every numeric cell.
- Parquet sources cache metadata/schema/file handles per source to avoid repeated metadata parsing.
- Research inference uses adaptive statistically defensible methods rather than blindly running expensive resampling on large bounded samples.
- Execution budgets now bound expensive quality, deep-statistics, anomaly, time-series, relationship, bootstrap, and distribution work by source scale and analysis mode.

### Correctness and statistical fidelity

- Full-stream missing counts, numeric counts, min/max, and moments are reused downstream with explicit provenance.
- Deep/Research shape summaries can reuse exact full-stream skewness/kurtosis while sample-dependent distribution fits, tests, intervals, and relationship diagnostics remain explicitly sample-scoped.
- Streaming ML-readiness, health, roles, quality, relationships, statistics, and anomaly results now distinguish exact/full-stream facts from bounded estimates.
- Categorical native profiling reports full-stream approximate cardinality/heavy-hitter provenance rather than pretending native sketches were row samples.
- Anderson-Darling normality diagnostics support modern and legacy SciPy contracts without relying on a fixed critical-value index.
- Large-source sampling regression tests cover periodic aliasing and native/fallback provenance contracts.

### Performance and scale validation

FrameVitals benchmarks are not advertised as universal speedups. Results depend on mode, backend, source shape, storage, and statistical fidelity.

Validated physical workloads include 1B, 2.5B, and 5B logical cells. On the 5B 500k × 10k workload, the native full-stream profiling kernel remains competitive while retaining full-stream native quantile sketches where the fallback may choose bounded row-sample quantiles for cost. All published stress workflows enforce no full materialization and correctness tolerances.

### Same-dataset release comparison vs 0.1.0

For an end-to-end release comparison, `v0.1.0` (`3da1432`) and the 0.2.0 release candidate (`05b11e5`) were run through the same public `framevitals.analyze(path, mode=..., artifacts=False)` API on one deterministic physical 10,000 × 64 CSV (640,000 cells). Both environments used Python 3.11 with the same pinned NumPy/pandas/SciPy/statsmodels/scikit-learn stack. FrameVitals 0.2.0 used its native Rust backend. Each version/mode received one warm-up followed by three measured runs in interleaved ABBAAB order; all 10,000 rows and all 64 columns were validated in every run.

| Mode | 0.1.0 median wall | 0.2.0 median wall | Speedup | 0.1.0 median peak RSS | 0.2.0 median peak RSS | Memory change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Quick | 1.644 s | 0.691 s | **2.38×** | 234.7 MB | 239.5 MB | 2.1% higher |
| Standard | 141.084 s | 0.839 s | **168.06×** | 2736.7 MB | 264.9 MB | **90.3% lower** |

The measured Quick ranges were 1.620–1.648 s for 0.1.0 and 0.686–0.692 s for 0.2.0. Standard ranges were 140.709–141.130 s for 0.1.0 and 0.823–0.841 s for 0.2.0. The large Standard improvement is a release-level architectural result: 0.2.0 replaces 0.1.0's fully materialized, unbounded heavy-statistics path with bounded/adaptive execution and native streaming. It should not be interpreted as a same-algorithm microbenchmark.

The machine-readable evidence is committed as `benchmarks/results/release_0.2.0_vs_0.1.0_10k_x64.json` from GitHub Actions run `32010158292`.

A separate 100,000 × 64 (6.4M-cell) stress run completed all Quick repetitions at a 5.097 s vs 1.447 s median (**3.52× faster**) and 406.1 MB vs 282.7 MB median peak RSS (**30.4% lower**) for 0.1.0 vs 0.2.0. The repeated Standard comparison intentionally has no formal speedup claim: the 30-minute workflow limit expired because completed 0.1.0 Standard passes took about 640–649 s and roughly 12.4 GB peak RSS each, while observed 0.2.0 Standard passes completed in about 1.58–2.06 s at roughly 303–308 MB. That incomplete stress evidence is GitHub Actions run `32008727705`.

### Packaging and compatibility

- Added Python 3.11, 3.12, and 3.13 core coverage plus Windows/macOS smoke testing.
- Added dedicated Arrow fallback, Rust/native, minimum-dependency, optional-feature, frontend, package-quality, documentation, and performance CI lanes.
- Added PEP 561 `py.typed` packaging support.
- Moved Excel, plotting, AI, Arrow, DuckDB, and other heavier capabilities behind explicit optional dependency groups where appropriate.
- Release CI verifies the GitHub tag, Python package version, `pyproject.toml`, Rust core version, and Rust bridge version before publication.
- Package CI verifies both fallback/native wheel contents and proves that pip prefers a compatible native wheel when both are available.

### Fixed

- Fixed periodic sampling aliasing caused by fixed evenly spaced global-row samples.
- Fixed projected streaming missingness/quality semantics so denominators describe the profiled projection rather than silently using the true source width.
- Fixed duplicate core recomputation in the streaming pipeline.
- Fixed stale sampling/native-categorical provenance assertions and public execution labels.
- Fixed SciPy Anderson-Darling compatibility and a pandas string-dtype selection deprecation.
- Fixed several optional-dependency/import boundaries so core/lightweight usage does not eagerly require plotting, AI, web, or Excel stacks.

## 0.1.0 - 2026-08-15

First public alpha release of FrameVitals.

### Added

- Installable `framevitals` package under `src/framevitals/`.
- Public `framevitals.analyze()` API for pandas DataFrames and supported dataset files.
- Public `framevitals.compare()` API for reference-vs-current drift analysis.
- `framevitals` command-line interface with analyze and compare commands.
- Data-health and ML-readiness scoring.
- Structural profiling and semantic column-role inference.
- Missingness, duplicate, cardinality, statistical, anomaly, target-aware, time-series, and text diagnostics.
- Drift comparison using PSI plus numeric/categorical statistical tests.
- Optional artifact generation and ML/AI/web dependency groups.
