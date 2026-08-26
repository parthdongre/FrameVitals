# Runtime configuration

FrameVitals can be configured from Python mappings or a TOML file passed with
`config=` / `--config`. Version 0.3 adds enforceable resource caps on top of the
existing adaptive execution modes.

```toml
[analysis]
preset = "exhaustive"
target = "churn"
artifacts = false

[resources]
workers = 4
max_sample_rows = 5000
max_relationship_pairs = 20
max_memory_heavy_parallelism = 1
max_streaming_profile_columns = 64

[modules]
modeling = false
ai = false
```

## Resource caps

The `max_*` settings are hard upper bounds. They only reduce work selected by a
mode; they never increase a mode's built-in sampling, relationship, streaming,
or parallelism budgets.

- `max_sample_rows` caps bounded row samples used by expensive diagnostics.
- `max_relationship_pairs` caps pairwise relationship/statistical work.
- `max_memory_heavy_parallelism` caps concurrent memory-heavy analysis tasks.
- `max_streaming_profile_columns` caps full-stream profiling width for streaming
  sources and can force deterministic schema projection before scanning.

`framevitals plan data.csv --config framevitals.toml` shows the resolved resource
policy and the effective execution budget before heavy analysis starts.

## Presets

The built-in presets are `quick`, `standard`, `deep`, `research`, `exhaustive`,
and `ci`. `exhaustive` currently maps to the established `research` execution
mode while the 0.x public mode names remain backward compatible.

## Precedence

Configuration currently resolves in this order, from lowest to highest:

1. FrameVitals defaults
2. explicit preset
3. config mapping/TOML values
4. explicit Python or CLI runtime arguments

Module toggles can be set under `[modules]`. Resource caps live under
`[resources]`; they are intentionally shared by `analyze()` and `plan()` so the
previewed budget matches execution.
