# Execution context

FrameVitals 0.3 introduces a per-run `AnalysisContext` as the shared state holder for
planning and, progressively, analysis execution. The goal is to stop independent
modules from rediscovering the same profile, roles, signals, samples, and planning
facts through repeated scans.

The context contains:

- resolved runtime configuration and enforceable resource policy;
- source identity and true dataset shape;
- authoritative facts such as profile, roles, signals, budget, and execution plan;
- a thread-safe reusable intermediate cache;
- named bounded samples that can be shared by downstream modules;
- a deterministic run seed.

## Cache semantics

`AnalysisContext.get_or_compute(key, factory)` computes a cache entry at most once
per context, even when scheduler threads request it concurrently. Cache state is
strictly per run; no process-global analysis cache is introduced.

Known intermediates can also be inserted with `cache_value()`. Authoritative facts
use `set_fact()` / `require_fact()`, which reject accidental replacement unless the
caller explicitly opts into `overwrite=True`.

## Sample reuse

`store_sample()` retains a named sample object for downstream modules. Context
metadata records only sample shape/provenance; it never serializes the raw sample.
That keeps `plan()` and future result provenance safe to inspect while allowing the
runtime to reuse the actual in-memory object.

## Current integration

`framevitals.plan()` now constructs one context and uses its cache for role
inference, dataset signals, execution-budget derivation, and planner construction.
The returned plan includes `execution_context` metadata with schema version, fact
names, cache statistics, and planning-sample provenance.

The next execution integration is to create the same context inside full materialized
and streaming analyses, populate it from their already-computed structural facts,
and make module schedulers request cached facts/samples from it instead of owning
parallel state.
