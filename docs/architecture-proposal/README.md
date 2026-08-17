# FrameVitals Architecture Proposal Archive

These documents were transferred from the former `docs/architecture-proposal` working branch into `develop/august` so the useful planning work can be preserved before branch cleanup.

They are **planning references, not the current public API or release contract**. Several originally proposed items have already evolved into implemented FrameVitals features, including result objects, source-aware execution, execution budgets, streaming/Arrow support, native Rust kernels, planning APIs, monitoring snapshots, quality gates, and capability-oriented optional dependencies.

For current behavior, prefer the maintained documentation in the parent `docs/` directory and the package code/tests.

## Preserved planning documents

- `IMPLEMENTATION_ROADMAP.md` — original staged implementation sequence and product philosophy.
- `BENCHMARK_ACCEPTANCE_PLAN.md` — correctness/performance/memory/backend acceptance ideas.
- `CAPABILITY_PACKS.md` — optional capability-pack and dependency-management design.
- `CLI_TUI_SPEC.md` — future interactive terminal design ideas.
- `DECISIONS.md` — accepted long-term product/architecture principles.
- `PROPOSED_MODULE_TREE.md` — proposed modular organization for future growth.

Generated PDF/LaTeX artifacts and the proposal-only build workflow were intentionally **not** transferred. Source planning content belongs in Git; generated planning artifacts do not need to remain on an active development branch.

## Core rule retained from the proposal

> FrameVitals should be exhaustive in capability, not exhaustive in what it runs by default.

That principle remains compatible with the current direction: the execution engine should choose the cheapest statistically defensible work based on applicability, scale, installed capabilities, backend availability, and requested analysis depth.
