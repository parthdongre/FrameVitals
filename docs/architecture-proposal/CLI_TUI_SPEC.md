# FrameVitals CLI / TUI Specification

This document defines the intended command-line and interactive terminal experience for future FrameVitals releases.

## Goals

The CLI should satisfy two very different users without forcing either into the other's workflow:

1. **Interactive user** — wants to explore configuration, install optional capabilities, select analyses, and inspect reports through a terminal UI.
2. **Automation user** — wants stable commands, exit codes, JSON output, and no prompts for scripts/CI.

The same underlying configuration and execution engine must power both experiences.

## Entry behavior

```bash
framevitals
```

If an interactive TTY is available and the TUI capability is installed, open the FrameVitals terminal application.

If the TUI is unavailable, show normal CLI help and a short message explaining how to install the interactive capability.

Explicit subcommands must never unexpectedly open the TUI:

```bash
framevitals analyze data.csv
framevitals compare reference.csv current.csv
framevitals validate data.csv --contract contract.json
```

## Main navigation

```text
FrameVitals

  Analyze Dataset
  Compare Datasets
  Validate Contract
  Reports & Snapshots
  Configuration
  Add-ons
  Models
  Doctor
  Help
  Exit
```

## Analysis screen

```text
Dataset: /Users/me/data/customers.csv
Target:  churn

Preset
( ) Quick   (*) Standard   ( ) Deep   ( ) Exhaustive   ( ) Custom

Analysis categories
[x] Structural profile
[x] Data quality
[x] Statistics
[x] Relationships
[x] ML readiness
[x] Target intelligence
[x] Drift / comparison       auto when reference exists
[x] Time-series analysis     auto when applicable
[x] Text analysis            auto when applicable
[x] Privacy / PII checks
[ ] Deep models              optional pack
[ ] AI interpretation        optional pack

Resources
Backend       Auto
Workers       4
Memory limit  4 GB
Time limit    120 s
GPU           Auto
Sampling      Adaptive

[ Show Plan ] [ Run Analysis ] [ Save Profile ]
```

## Plan screen

Before expensive runs, users should be able to see what FrameVitals intends to execute.

```text
Execution Plan

RUN   structural_profile        essential     ~low
RUN   missingness_analysis      essential     ~low
RUN   relationship_matrix       high          ~medium
RUN   anomaly_ensemble          high          ~medium
SKIP  tcn_temporal_anomaly      not installed
SKIP  text_embeddings           no applicable text column
SKIP  target_leakage            no target selected

Estimated class: medium
Expected peak memory: < 2 GB

[ Run ] [ Configure ] [ Back ]
```

Every skipped item should have an explicit reason.

## Add-on manager

Optional functionality must be easy to discover and install from the TUI.

```text
Add-ons

Core
[x] Core analysis              Built in
[x] Contracts                  Built in

Optional
[x] Visual reports             Installed     Enabled
[ ] Excel support              Not installed [ Install ]
[x] ML diagnostics             Installed     Enabled
[ ] Deep models                Not installed [ Install ]
[ ] Advanced text/NLP          Not installed [ Install ]
[ ] Polars backend             Not installed [ Install ]
[ ] Arrow backend              Not installed [ Install ]
[ ] SQL adapters               Not installed [ Install ]
[ ] Cloud filesystems          Not installed [ Install ]

Space  Toggle enabled state
Enter  Install / Open details
R      Remove
```

### Installation flow

Selecting `Install` should show exactly what will happen:

```text
Install: Deep Models

Provides
- Autoencoder anomaly detector
- DeepSVDD anomaly detector
- Optional temporal CNN/TCN diagnostics

Python packages to install
  framevitals[deep]

Estimated package download: 220 MB
Model downloads: none until a model is explicitly selected

Command
  /path/to/python -m pip install "framevitals[deep]"

[ Install ] [ Cancel ]
```

Rules:
- Never install implicitly during `import framevitals`.
- Never install from a normal non-interactive analysis command.
- Interactive install requires confirmation.
- Show which Python environment will be modified.
- Display installation errors instead of hiding them.
- Allow user to copy the command and run it manually.

## Model manager

Python add-ons and model weights are different resources and should be managed separately.

```text
Models

Semantic Type Classifier
[ ] semantic-types-small       24 MB       [ Download ]

Text
[ ] text-embeddings-small      90 MB       [ Download ]

Time Series
[ ] temporal-anomaly-tiny      18 MB       [ Download ]

Anomaly
[ ] tabular-ae-reference       12 MB       [ Download ]
```

Model registry metadata should include:
- model ID
- model version
- compatible FrameVitals versions
- task
- file size
- checksum
- license
- source
- framework/runtime requirement
- expected hardware

Downloaded weights should be cached in an OS-appropriate FrameVitals data directory, not in the working project directory.

## Configuration screen

Configuration should be editable interactively but stored in a human-readable file.

Suggested conceptual config:

```toml
[analysis]
preset = "standard"
statistics = true
relationships = true
ml_readiness = true
deep_models = false
ai_interpretation = false

[resources]
workers = 4
max_memory = "4GB"
max_time_seconds = 120
gpu = "auto"
sampling = "adaptive"

[reporting]
terminal = true
html = false
json = false

[privacy]
pii_detection = true
show_raw_sensitive_examples = false
```

## Configuration precedence

Highest priority wins:

1. Explicit Python arguments / command flags
2. Project configuration (`framevitals.toml`)
3. User configuration
4. Environment variables
5. Selected preset
6. Built-in defaults

`framevitals config explain` should show where each effective value came from.

## Scriptable command design

Interactive actions should have deterministic command equivalents.

```bash
# analysis
framevitals analyze data.csv --preset deep
framevitals analyze data.csv --target churn --html report.html --json report.json
framevitals plan data.csv --preset exhaustive --explain

# configuration
framevitals config show
framevitals config set analysis.deep_models true
framevitals config profile use laptop
framevitals config explain

# add-ons
framevitals addons list
framevitals addons install deep
framevitals addons enable deep
framevitals addons disable deep
framevitals addons remove deep

# models
framevitals models list
framevitals models install semantic-types-small
framevitals models remove semantic-types-small

# environment diagnostics
framevitals doctor
framevitals doctor --json
```

## Exit codes

Suggested stable policy:
- `0` — command completed / validation passed
- `1` — validation or configured quality gate failed
- `2` — invalid arguments/configuration
- `3` — input/data loading failure
- `4` — optional capability unavailable
- `5` — internal execution failure

Exact values can change before stabilization, but the policy should become documented and tested before 1.0.

## Non-interactive safety

When stdin/stdout is not interactive:
- never prompt
- never install packages
- never download models unless an explicit install command was invoked
- emit deterministic output
- respect `--json`/`--quiet`
- use stable exit codes

## Doctor screen

`framevitals doctor` should answer common environment questions:

```text
FrameVitals 0.x
Python        3.13.7
Platform      macOS arm64
Core          OK
ML pack       Installed
Deep pack     Missing
Polars        Missing
PyArrow       Installed
GPU runtime   Not detected
Cache         /Users/me/Library/Caches/framevitals
Config        ~/.config/framevitals/config.toml

No blocking problems detected.
```

## UX principle

The TUI should hide accidental complexity, not analytical detail. Users should be able to make common decisions with a toggle or button while still being able to inspect:
- exactly what analysis is being run
- why it was selected
- what it costs
- which optional dependency/model it uses
- what command/change an install button performs
