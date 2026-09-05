# Execution planning

FrameVitals 0.3 introduces a versioned planner contract behind `framevitals.plan()`.
The planner separates three questions that were previously mixed together:

1. which analyses are applicable to the observed dataset signals;
2. which runtime modules are disabled by mode or explicit configuration;
3. which modules are expected to run, are not applicable, or remain conditional.

A plan now includes `selection.planner_schema_version` and structured execution-module
decisions under `selection.execution_modules.decisions`.

```python
plan = framevitals.plan(df, mode="standard", target="churn")

print(plan.planner_schema_version)
print(plan.module_decisions["anomaly_detection"])
```

Each module decision contains:

- `status`: `run`, `conditional`, `not_applicable`, `disabled_by_mode`, or
  `disabled_by_config`;
- `reason`: a human-readable explanation;
- `resource_class`: a stable coarse cost category such as `bounded_cpu` or
  `memory_heavy`;
- `depends_on`: runtime module dependencies.

The compatibility fields `execution_modules.disabled` and
`execution_modules.enabled` remain available. `effective_disabled` additionally
shows the union of explicit configuration and built-in mode policy.

## Planner ownership in 0.3

The built-in mode-to-module policy is now centralized in `framevitals.planner` and
is also used by the public `analyze()` dispatcher. This removes one source of drift
between planning and execution while keeping current pipeline behavior stable.

Signal-based `not_applicable` decisions are currently explanatory: the next planner
integration step is for both the materialized and streaming schedulers to consume
those decisions directly after their shared structural facts are available. This is
intentional so FrameVitals does not add a second pre-analysis scan just to make the
planner authoritative.
