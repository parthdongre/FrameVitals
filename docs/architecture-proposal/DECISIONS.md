# FrameVitals Architecture Decisions

This file records decisions that were explicitly accepted during architecture planning so future implementation does not repeatedly reopen the same product questions without new evidence.

These are direction-setting principles, not a frozen public API contract. Current package behavior and stability guarantees are defined by the maintained code, tests, and release documentation.

## D-001 — Exhaustive capability, selective execution

**Status:** Accepted

FrameVitals may aim for a very broad and eventually exhaustive catalog of tabular-data diagnostics. A normal analysis must not run everything blindly.

Default execution is selected by:
- data applicability
- preset/configuration
- installed and enabled capabilities
- resource budget
- analysis dependencies

`exhaustive` means every applicable installed analysis permitted by the active resource policy.

---

## D-002 — Interactive terminal UI plus scriptable CLI

**Status:** Accepted direction

Running `framevitals` with no explicit subcommand may eventually open an interactive terminal application when that capability is installed.

Explicit commands such as `framevitals analyze`, `compare`, and `validate` remain deterministic and suitable for scripts/CI.

The TUI and CLI should share the same configuration and execution engine.

---

## D-003 — Optional features use install actions/toggles

**Status:** Accepted direction

Optional capabilities should be discoverable rather than forcing users to research dependency groups manually.

Any interactive installation action must still show the exact command/environment being modified and request confirmation.

Installed and enabled are separate states.

---

## D-004 — No implicit package/model downloads

**Status:** Accepted

FrameVitals never installs Python packages or downloads model weights during normal import or ordinary analysis.

Downloads occur only through explicit add-on/model installation actions or equivalent dedicated commands.

---

## D-005 — Deep learning is optional and task-specific

**Status:** Accepted

ML/DL is used only where it adds analytical value, for example:
- nonlinear multivariate anomaly detection
- temporal CNN/TCN diagnostics for ordered time-series windows
- optional semantic-type second opinion
- embedding-based text drift
- nonlinear learnability diagnostics

CNNs are not applied to generic unordered tabular rows merely because they are available.

Deterministic/statistical fallbacks remain first-class.

---

## D-006 — FrameVitals is not an AutoML platform

**Status:** Accepted

Baseline models and nonlinear diagnostic models can measure learnability, leakage, feature behavior, or explainability. The product does not optimize around training/deploying a production model fleet.

The central question remains: **what is true about this dataset, what is wrong, what changed, and what should the user investigate next?**

---

## D-007 — Result/report UX precedes algorithm count

**Status:** Accepted

Stable result objects/schema, normalized findings, full JSON output, good terminal rendering, and useful reports are higher priority than continuously adding isolated algorithms.

New analysis is only valuable when users can consume and trust the result.

---

## D-008 — One reusable analysis context / exact-once facts

**Status:** Accepted direction

Modules should consume reusable facts from the execution context rather than independently rescanning the same dataset for missingness, dtypes, semantic roles, samples, moments, correlations, and related facts.

When the source pass has already established an exact sufficient statistic, downstream diagnostics should reuse it rather than replace it with a bounded-sample estimate.

---

## D-009 — Cleaning is plan-first

**Status:** Accepted direction

FrameVitals may recommend/simulate cleaning operations. It should not silently mutate user data.

Preferred workflow:
1. detect issue
2. propose cleaning action
3. estimate/simulate impact when possible
4. user explicitly applies approved transformations
5. record an audit trail

---

## D-010 — Public project website is separate from analysis runtime

**Status:** Accepted

The public website is primarily for product explanation, demos, installation, documentation, and benchmark/transparency information.

It does not need to become a dataset-upload service. The website and generated reports can share a coherent visual language while remaining separate products.

---

## D-011 — Planning material is preserved but not release contract

**Status:** Accepted

Architecture proposal material is preserved under `docs/architecture-proposal/` in the integration history. It should not be treated as the current release contract.

Implementation changes still land through normal development/review and must earn their place through correctness, resource, compatibility, and benchmark evidence.
