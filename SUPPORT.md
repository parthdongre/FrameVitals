# FrameVitals support

FrameVitals is alpha software and the fastest way to fix a problem is a small,
reproducible report that shows **what input/source path was used, which optional
capabilities were installed, and what FrameVitals actually executed**.

## Before opening an issue

1. Reproduce the problem on the newest compatible FrameVitals release or the relevant
   development branch.
2. Reduce the input to the smallest dataset that still reproduces the behavior.
3. Check whether the same call behaves differently with a pandas DataFrame versus the
   original source adapter.
4. Include execution/source metadata when the problem involves memory, sampling,
   streaming, or performance.

Useful commands:

```bash
framevitals --version
framevitals system-info --no-probe-gpu --format json
framevitals inspect path/to/dataset.parquet --format json
```

If the problem specifically involves CUDA/GPU probing, run `framevitals system-info`
without `--no-probe-gpu` as well.

## What to include in a bug report

Please provide:

- FrameVitals version or commit SHA;
- Python version and operating system;
- the exact command or Python call;
- the smallest reproducible input shape/schema;
- which optional extras are installed (`arrow`, `duckdb`, `excel`, `plot`, `ml`, `ai`,
  or `web`);
- `framevitals system-info --no-probe-gpu --format json` output when relevant;
- `framevitals inspect ... --format json` output for source/streaming issues;
- the complete exception traceback;
- expected behavior and actual behavior.

For a focused diagnostic, include its execution metadata when possible:

```python
result = fv.statistics(data, mode="quick")
print(result.execution)
```

This tells maintainers whether the operation sampled, streamed, or fully materialized
its source.

## Reproduction datasets

Do **not** upload confidential, personal, regulated, proprietary, or customer data to a
public issue. Prefer one of these:

- a tiny synthetic DataFrame;
- a short CSV created specifically for the reproduction;
- code that deterministically generates the failing data pattern;
- schema/source metadata with sensitive values removed.

## Performance reports

For performance regressions, include:

- dataset rows/columns and approximate file size;
- source format and whether `inspect_source()` reports streaming/projection support;
- analysis mode;
- `result.execution` when available;
- wall-clock time and peak memory if measured;
- whether `FRAMEVITALS_BACKEND` was configured explicitly;
- native-core availability from `system-info`.

The repository also contains reproducible benchmark and catastrophic-regression
workflows under `benchmarks/` and `.github/workflows/`.

## Feature requests

A strong feature request explains the workflow problem before proposing a new module.
FrameVitals deliberately prefers a small analyze → compare → validate → gate → monitor
surface over accumulating unrelated analytics features.

For source integrations, first check whether the producer already supports the Arrow C
Stream / PyCapsule protocol; a standard interoperability boundary is preferable to a
library-specific dependency when it preserves the required semantics.

## Security issues

Do not report vulnerabilities, credential exposure, or sensitive-data security issues
through a normal public support thread. Follow the repository's `SECURITY.md` process.

## General questions

Questions that can become durable documentation are welcome. When possible, include a
small code example and the desired outcome so the answer can be turned into a docs or
example improvement for future users.
