# FrameVitals Implementation Roadmap

This document turns the future architecture proposal into a practical sequence of pull requests. It is a planning reference, not a promise that every item ships in the named release.

## Operating rule

FrameVitals should become **exhaustive in capability and selective in execution**. New capabilities should be added behind stable interfaces, configuration, applicability checks, and resource budgets rather than wired directly into one ever-growing pipeline.

## Build order

### PR 1 — Result model foundation
**Priority:** P0

Create a stable internal/public result layer without breaking the current dictionary-returning API.

Proposed modules:
- `src/framevitals/result.py`
- `src/framevitals/findings.py`
- `src/framevitals/metadata.py`

Deliverables:
- `AnalysisResult` or an internal result object with mapping compatibility.
- Normalized `Finding` structure: code, title, severity, confidence, evidence, affected columns, recommendation, method.
- `to_dict()` and JSON-safe serialization.
- Existing `fv.analyze()` output remains backward compatible during migration.

Acceptance criteria:
- Existing tests still pass.
- Old dictionary access remains available.
- Result schema receives an explicit version.

---

### PR 2 — Full report export and terminal renderer
**Priority:** P0

Fix the current gap between the rich Python result and the summary-only CLI output.

Proposed modules:
- `src/framevitals/reporting/terminal.py`
- `src/framevitals/reporting/json.py`

Deliverables:
- Human-friendly terminal summary.
- Full JSON export from CLI.
- Clear distinction between terminal summary and machine-readable full result.
- Stable exit behavior.

Acceptance criteria:
- `framevitals analyze data.csv --output report.json` can intentionally write the full report.
- Terminal output is concise and does not dump enormous nested JSON by default.

---

### PR 3 — Standalone HTML report
**Priority:** P0

Proposed modules:
- `src/framevitals/reporting/html.py`
- `src/framevitals/reporting/assets.py`

Deliverables:
- Self-contained HTML report requiring no server.
- Overview, health, findings, missingness, column profiles, relationships, anomalies, target intelligence, drift when available, and recommendations.
- Deterministic report generation from an existing result object.

Acceptance criteria:
- Report works offline in a browser.
- No user dataset is uploaded anywhere.
- HTML generation does not re-run analysis.

---

### PR 4 — Configuration and presets
**Priority:** P0

Proposed modules:
- `src/framevitals/config.py`
- `src/framevitals/config_presets.py`
- `src/framevitals/config_profiles.py`

Deliverables:
- Typed configuration object.
- Presets: `quick`, `standard`, `deep`, `exhaustive`, `custom`.
- Resource policy: workers, memory budget, time budget, sampling policy, GPU preference.
- Analysis-category toggles.
- Config precedence rules.

Suggested precedence:
1. explicit Python/CLI arguments
2. project config
3. user config
4. environment variables
5. preset defaults

Acceptance criteria:
- Same config can be used from Python and CLI.
- Config validation produces useful errors.

---

### PR 5 — AnalysisContext and reusable cache
**Priority:** P0

Proposed modules:
- `src/framevitals/execution/context.py`
- `src/framevitals/execution/cache.py`
- `src/framevitals/execution/sampling.py`

Deliverables:
- One context per run containing loaded data/backend, shape, profile, roles, fingerprints, reusable samples, configuration, seed, installed capabilities, and cached intermediates.
- Modules request reusable facts from the context rather than rescanning the dataset.

Acceptance criteria:
- Current outputs remain equivalent for deterministic analyses.
- Benchmarks demonstrate fewer full-data scans on representative datasets.

---

### PR 6 — Planner-controlled execution
**Priority:** P0

Evolve `analysis_inventory.py` + `analysis_selector.py` into a real execution planner.

Proposed modules:
- `src/framevitals/planner.py`
- `src/framevitals/execution/budget.py`

Deliverables:
- Registry entries declare inputs, applicability, cost, dependencies, outputs, optional pack, and resource class.
- Planner produces an explainable execution plan.
- `framevitals plan data.csv --explain` shows what will run and why.
- Exhaustive mode means every applicable installed analysis that fits the configured policy.

Acceptance criteria:
- The planner actually controls execution rather than only describing it.
- Skipped analyses record a reason.

---

### PR 7 — Interactive CLI/TUI foundation
**Priority:** P1

Proposed modules:
- `src/framevitals/cli_tui.py`
- `src/framevitals/doctor.py`

Deliverables:
- Running `framevitals` without a subcommand opens the interactive interface when the TUI is available.
- Analyze, Compare, Validate, Reports, Configuration, Add-ons, Models, Doctor.
- Traditional subcommands remain fully usable for scripts and CI.

Acceptance criteria:
- No TUI dependency is imported during a normal library import.
- Non-interactive behavior remains stable.

---

### PR 8 — Add-on manager and capability registry
**Priority:** P1

Proposed modules:
- `src/framevitals/addons.py`
- `src/framevitals/capabilities.py`
- `src/framevitals/models/registry.py`

Deliverables:
- Installed/enabled are separate states.
- Discover whether optional dependencies are installed.
- TUI install/remove/toggle actions.
- Scriptable equivalents: `addons list/install/remove/enable/disable`.
- Exact command displayed before modifying the active environment.

Acceptance criteria:
- Importing FrameVitals never installs anything.
- Installation requires explicit confirmation in interactive mode.

---

### PR 9 — Semantic types and deeper profiling
**Priority:** P1

Proposed modules:
- `src/framevitals/semantic_types.py`
- `src/framevitals/quality/keys.py`
- `src/framevitals/quality/missingness.py`

Deliverables:
- Stronger semantic types: ID, UUID, email, URL, IP, phone-like, currency, percentage, geospatial-like, date/time, free text, code/category.
- Candidate primary keys and composite-key hints.
- Missingness patterns and co-occurrence.
- Categorical normalization suggestions.
- Optional small character model can later act as a second opinion when rule confidence is low.

---

### PR 10 — Target intelligence and relationship engine
**Priority:** P1

Proposed modules:
- `src/framevitals/target_intelligence.py`
- `src/framevitals/relationships.py`
- `src/framevitals/ml/label_quality.py`

Deliverables:
- Target/task inference.
- Class imbalance and target quality.
- Proxy, temporal, and split leakage.
- Numeric↔numeric, categorical↔categorical, numeric↔categorical relationships.
- Feature stability/redundancy signals.

Optional model support:
- nonlinear baseline as a diagnostic, not AutoML.

---

### PR 11 — Drift, snapshots, and monitoring foundation
**Priority:** P1

Proposed modules:
- `src/framevitals/drift/`
- `src/framevitals/monitoring/snapshot.py`
- `src/framevitals/monitoring/store.py`

Deliverables:
- Pluggable drift methods and automatic method selection.
- Drift severity and confidence/evidence.
- Schema diff.
- Result snapshots and local history.
- Compare current result against a baseline snapshot.

---

### PR 12 — Scale/backends and first optional model pack
**Priority:** P2

Proposed modules:
- `src/framevitals/backends/`
- `src/framevitals/io/`
- `src/framevitals/models/deep_anomaly.py`
- `src/framevitals/models/runtime.py`

Deliverables:
- Polars/PyArrow/Narwhals strategy.
- Parquet and chunked/streaming paths.
- Large-data sampling/budget behavior.
- Optional anomaly pack: autoencoder and/or DeepSVDD alongside classical detectors.
- Optional GPU selection only when useful and installed.

Acceptance criteria:
- Core install stays usable without deep-learning frameworks.
- Model pack failure never removes deterministic fallback analysis.

## Parallel workstreams

The public project website can be developed in parallel because it should initially use curated/precomputed demo data rather than depend on the full backend roadmap.

Recommended parallel streams:
- **Core:** PR 1–6
- **Experience:** PR 2–3, 7–8
- **Analysis depth:** PR 9–11
- **Scale/models:** PR 12+
- **Website:** independent visual/demo frontend with shared report design language

## Release philosophy

Do not ship a release because the project gained a certain number of algorithms. Ship when a coherent user workflow becomes materially better.

Examples:
- result/report release
- configuration/planner release
- interactive/add-on release
- monitoring release
- scale/backend release
