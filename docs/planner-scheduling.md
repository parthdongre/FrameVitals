# Planner scheduling contract

FrameVitals 0.3 planner output is structured so it can be consumed by a runtime
scheduler rather than only displayed to users.

## Dependency blocking

Runtime modules declare `depends_on`. After applicability and configuration rules
are evaluated, the planner propagates blocked dependencies downstream. A dependent
module is changed to `not_applicable` and receives a `blocked_by` list plus a reason.

For example, explicitly disabling `target_intelligence` in a research run also blocks
`modeling`, which then blocks `explainability`. The planner will not advertise work
that cannot satisfy its declared prerequisites.

## Execution stages

`selection.execution_modules.execution_stages` contains topologically ordered stages.
Each stage contains:

- `stage`: zero-based stage index;
- `modules`: modules whose dependencies are satisfied by earlier stages;
- `resource_classes`: coarse resource classes represented by those modules.

`runnable_modules` is the flattened stage order and includes both `run` and
`conditional` decisions. Conditional modules still require their runtime condition
to become true—for example, explainability requires modeling to produce a suitable
winner.

This is the scheduling interface that the materialized and streaming executors can
adopt incrementally while preserving the existing result schema.
