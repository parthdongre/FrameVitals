# Proposed Future Module Tree

This is a destination architecture, **not** a request to rename/move every current module immediately. Existing public imports should remain compatible while internal organization evolves gradually.

```text
src/framevitals/
├── __init__.py
├── api.py
├── result.py
├── findings.py
├── metadata.py
├── errors.py
├── config.py
├── config_presets.py
├── config_profiles.py
├── capabilities.py
├── addons.py
├── doctor.py
│
├── execution/
│   ├── __init__.py
│   ├── context.py
│   ├── planner.py
│   ├── inventory.py
│   ├── budget.py
│   ├── cache.py
│   ├── sampling.py
│   └── scheduler.py
│
├── io/
│   ├── __init__.py
│   ├── loader.py
│   ├── csv.py
│   ├── excel.py
│   ├── json.py
│   ├── parquet.py
│   ├── sql.py
│   └── cloud.py
│
├── backends/
│   ├── __init__.py
│   ├── base.py
│   ├── pandas.py
│   ├── polars.py
│   ├── arrow.py
│   └── narwhals_adapter.py
│
├── profiling/
│   ├── __init__.py
│   ├── structural.py
│   ├── semantic_types.py
│   ├── column_roles.py
│   ├── dataset_signals.py
│   ├── keys.py
│   ├── missingness.py
│   └── freshness.py
│
├── quality/
│   ├── __init__.py
│   ├── health.py
│   ├── duplicates.py
│   ├── consistency.py
│   ├── ranges.py
│   ├── categories.py
│   └── pii.py
│
├── contracts/
│   ├── __init__.py
│   ├── infer.py
│   ├── validate.py
│   ├── rules.py
│   ├── policy.py
│   ├── diff.py
│   └── schema.py
│
├── statistics/
│   ├── __init__.py
│   ├── descriptive.py
│   ├── distributions.py
│   ├── robust.py
│   ├── testing.py
│   ├── relationships.py
│   ├── multicollinearity.py
│   └── segments.py
│
├── anomaly/
│   ├── __init__.py
│   ├── ensemble.py
│   ├── univariate.py
│   ├── classical.py
│   └── scoring.py
│
├── drift/
│   ├── __init__.py
│   ├── compare.py
│   ├── numeric.py
│   ├── categorical.py
│   ├── text.py
│   ├── schema.py
│   ├── methods.py
│   └── selection.py
│
├── target/
│   ├── __init__.py
│   ├── intelligence.py
│   ├── task.py
│   ├── leakage.py
│   ├── imbalance.py
│   ├── label_quality.py
│   └── candidates.py
│
├── ml/
│   ├── __init__.py
│   ├── readiness.py
│   ├── preprocessing.py
│   ├── baseline.py
│   ├── leaderboard.py
│   ├── diagnostics.py
│   ├── feature_importance.py
│   └── explainability.py
│
├── time_series/
│   ├── __init__.py
│   ├── detection.py
│   ├── quality.py
│   ├── diagnostics.py
│   ├── seasonality.py
│   ├── change_points.py
│   └── anomaly.py
│
├── text/
│   ├── __init__.py
│   ├── profile.py
│   ├── quality.py
│   ├── duplication.py
│   ├── pii.py
│   └── drift.py
│
├── cleaning/
│   ├── __init__.py
│   ├── plan.py
│   ├── suggestions.py
│   ├── simulate.py
│   ├── apply.py
│   └── audit.py
│
├── monitoring/
│   ├── __init__.py
│   ├── snapshot.py
│   ├── store.py
│   ├── history.py
│   └── policy.py
│
├── models/
│   ├── __init__.py
│   ├── registry.py
│   ├── manifest.py
│   ├── runtime.py
│   ├── semantic_types.py
│   ├── deep_anomaly.py
│   ├── temporal.py
│   └── embeddings.py
│
├── reporting/
│   ├── __init__.py
│   ├── terminal.py
│   ├── json.py
│   ├── html.py
│   ├── pdf.py
│   ├── notebook.py
│   ├── charts.py
│   ├── compare_html.py
│   └── privacy.py
│
├── plugins/
│   ├── __init__.py
│   ├── registry.py
│   ├── checks.py
│   ├── analyzers.py
│   └── hooks.py
│
├── ai/
│   ├── __init__.py
│   ├── insights.py
│   ├── agent.py
│   ├── tools.py
│   └── brief.py
│
├── cli.py
└── cli_tui.py
```

## Migration philosophy

Do **not** perform a giant move-only refactor immediately. Prefer:

1. introduce a new namespace when implementing a new coherent capability
2. move an old module only when it is actively being changed or causes architectural friction
3. keep compatibility shims for public imports where necessary
4. delete shims only through documented deprecation cycles

## Mapping from important current modules

| Current module | Future home/direction |
|---|---|
| `pipeline.py` | gradually replaced/orchestrated by `execution/context.py` + `execution/planner.py` |
| `analysis_inventory.py` | `execution/inventory.py` |
| `analysis_selector.py` | `execution/planner.py` |
| `profiler.py` | `profiling/structural.py` |
| `column_roles.py` | `profiling/column_roles.py` + semantic typing layer |
| `dataset_signals.py` | `profiling/dataset_signals.py` |
| `health_score.py` | `quality/health.py` |
| `contracts.py` | split gradually into `contracts/` package |
| `drift_analysis.py` | split gradually into `drift/` package |
| `anomaly_ensemble.py` | `anomaly/ensemble.py` |
| `ml_readiness.py` | `ml/readiness.py` |
| `target_analyzer.py` | `target/intelligence.py` |
| `target_leakage.py` | `target/leakage.py` |
| `time_series.py` | split into `time_series/` package |
| `text_profile.py` | `text/profile.py` |
| `cleaner.py` | replaced by plan/simulate/apply/audit workflow in `cleaning/` |
| `visualizer.py` | `reporting/charts.py` |
| PDF/report modules | `reporting/` |
| `ai_agent.py`, `ai_insights.py`, `agent_tools.py`, `agent_brief.py` | isolated under optional `ai/` namespace |

## Dependency direction

A useful dependency rule is:

```text
api/result/reporting
        ↓
execution planner/context
        ↓
analysis domains
        ↓
profiling/backends/config
```

Analysis-domain modules should **not** import the CLI/TUI or website code. Reporting should consume result structures rather than cause analyses to rerun. Optional AI should consume structured findings/results rather than become a required dependency of deterministic modules.

## Plugin boundary

Long term, third-party analysis plugins should register through a stable plugin API instead of monkey-patching the pipeline.

A plugin should declare:
- ID/name/version
- supported FrameVitals plugin API version
- applicability predicate
- required inputs/context facts
- resource class
- optional dependency requirements
- output/finding schema
- whether failure is fatal or optional

This allows FrameVitals to be exhaustive without requiring every niche analysis to live in the core repository forever.
