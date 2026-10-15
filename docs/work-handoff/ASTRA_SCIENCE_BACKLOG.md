# Phase 2 status

The original backlog below is preserved as historical problem framing. Phase 2 results are in [the scientific audit](../audits/ASTRA_SCIENTIFIC_AUDIT.md), [methods](../research/ASTRA_MATHEMATICAL_METHODS.md), [algorithm evaluation](../research/ASTRA_ALGORITHM_EVALUATION.md), and [benchmarks](../benchmarks/ASTRA_PERFORMANCE_REPORT.md).

Addressed: spectral population/scale/condition errors, bounded full nonfinite checks, heuristic score interpretation, graph metric provenance and Laplacian definition, drift multiplicity/contingency assumptions, explicit group/time leaderboard splits, relational typed-key/population metrics, quantile relative-error representative and merge configuration, streaming extreme initialization, regular time-axis gating, alias/cycle distinction and runtime observation limits.

Remaining: calibrated domain scores and threshold false-alarm rates; HLL/heavy-hitter/reservoir distribution studies; independent nested model evaluation and higher-level group/time API routing; architecture-specific sample accuracy; structural graph rewiring sensitivity; extreme-scale tensor/checkpoint drift norms; VIF sentinel migration; negative-cost shortest-path API; recent-paper SSSP implementation reproduction; full statistical coverage simulations. Passing tests does not close these scientific questions.

---

# Astra science backlog

Start from the reviewed Phase 1 branch head pinned in PR #66. Do not start again from main or dev. Fixtures: `tests/fixtures/science/cases.json`. The following are validation tasks, not claims that every listed algorithm is defective. Priorities describe interpretation risk; Phase 1 did not change scientific formulas.

## P0 — Decimation and spectral denominators

- **File/functions:** `analysis/tensor.py::_matrix_sketch`, `tensor_metrics`, `analyze_tensor`.
- **Problem:** A deterministic submatrix can erase periodic structure. Approximate stable-rank normalization uses analyzed dimensions; a user can confuse this with full-matrix effective dimensionality.
- **Reproduction/fixture:** `tensor_strided_rank`: `np.diag(np.tile([1.,0.,1.,0.], 600))`; run Prism quick, inspect `tensor.matrix.original_shape`, `analyzed_shape`, `method`, `stable_rank_ratio`.
- **Expected mathematical behavior:** Full diagonal rank and stable rank are analytically 1200, ratio 0.5 against minimum original dimension. A sketch result must quantify its distinct population and uncertainty; it must not imply an unbiased full-matrix rank estimate.
- **Actual implementation:** The 2400-square matrix is strided by four, selecting ones and producing a 600-square identity submatrix. Randomized SVD operates on that submatrix; its stable-rank ratio can approach 1.0 relative to the analyzed minimum dimension. The engine exposes approximate metadata but also applies findings/score thresholds to approximate ratios.
- **Risk:** Misleading representation-collapse findings or cross-run comparisons.
- **Validation:** Use diagonal/block-diagonal/permuted analytic oracles; test strided aliasing; compare full/randomized/subsampled spectra and denominator definitions. Phase 1 only adds up to 64 existing singular values for plotting; their calculation is unchanged.

## P0 — Nonfinite detection and sampling coverage

- **File/functions:** `analysis/tensor.py::_numpy_sample`, `tensor_metrics`; analogous weight samplers in `analysis/model.py`, `safetensors.py`.
- **Problem:** Stride samples can omit isolated NaN/Inf values. Sample absence must not become a full-source safety guarantee.
- **Reproduction/fixture:** `tensor_omitted_tail_nonfinite`: 500001 zero values, NaN at index 499999; Prism quick.
- **Expected behavior:** Counts/fractions must state their sampled population, distinguish observations from population estimates, and avoid certifying complete finiteness unless all values are checked.
- **Actual implementation:** Quick mode samples at most 50000 strided values; the selected positions miss 499999. The sample reports zero NaNs while `sample_fraction` is below one. Existing findings use sample fractions.
- **Risk:** A numerically corrupt tensor/model looks healthy in a sample.
- **Validation:** Adversarial placement, varied lengths/seeds, exact streaming nonfinite counts versus bounded distribution samples. Preserve budgets and never silently allocate an entire huge model.

## P0 — Domain-dependent health scoring

- **File/functions:** `health_score.py::calculate_health_score`; `analysis/{graph,tensor,nested,model,document,relational}.py::analyze_*`.
- **Problem:** Scores use hand-selected penalties; low rank, sparsity, disconnectedness, duplication or empty extracted text can be valid structure or extraction limits rather than data faults.
- **Reproduction/fixture:** `health_context_dependence`, plus blank PDF in `test_blank_pdf_is_not_reported_as_fully_understood`.
- **Expected behavior:** Explain score basis, applicability and uncertainty; use context-sensitive findings and validated calibration if risk language is desired.
- **Actual implementation:** Domain-specific score deductions and labels are deterministic heuristics. Structured score metadata and the new browser explicitly label them heuristic; no cross-domain calibration exists.
- **Risk:** Comparing scores across unrelated domains or treating an OCR limitation as substantive document quality.
- **Validation:** Hand-labeled domain fixtures and monotonicity/degeneracy tests; define what a score claims before changing weights. Preserve old score fields or explicitly version changed methodology.

## P1 — Metric-level graph provenance and paths

- **File/functions:** `analysis/graph.py::_component_summary`, path/centrality/spectral helpers, `analyze_graph`; `analysis/change.py::compare_graphs`.
- **Problem:** One aggregate sampling flag cannot describe exact topology counts plus sampled centrality/path work.
- **Reproduction/fixture:** `graph_single_component_sampling`, `graph_directed_reachability`, `graph_negative_edge`; existing `test_diagnostic_safety.py` negative-edge cases.
- **Expected behavior:** Exact counts and sampled estimators should have separate coverage/method metadata. Directed path metrics need an explicit reachable-pair population; Dijkstra must never see negative weights. Negative-cycle behavior must be defined.
- **Actual implementation:** Aggregate graph execution marks `sampled=True` even for tiny graphs. Per-section budgets/methods exist. Full edge-weight validation and bounded negative-weight handling already have safety regressions; scientific estimator accuracy remains unvalidated.
- **Risk:** False precision in average distance, sampled connectivity or cross-run drift.
- **Validation:** Analytic path/star/cycle/disconnected/directed graphs; compare exact NetworkX oracles on small graphs; seeded sampling error curves; negative-cycle fixtures. Do not add a fashionable shortest-path algorithm without applicable assumptions and measured benefit.

## P1 — Drift statistics and multiple comparisons

- **File/functions:** `drift_analysis.py`, `analysis/change.py`, `analysis/document.py::compare_documents`, `analysis/nested.py::compare_nested`, `analysis/relational.py::compare_relational`.
- **Problem:** Source families use different changes/severity rules; small samples and degenerate distributions need explicit interpretation.
- **Reproduction/fixture:** `drift_small_sample` and existing compare/Tide tests.
- **Expected behavior:** Invariance under harmless row reordering where appropriate, explicit effect sizes, consistent missing/category support, small-sample handling and multiple-testing policy.
- **Actual implementation:** Different bounded diagnostics and heuristic severity rules; tests establish execution/serialization and some invariants, not global false-positive control.
- **Risk:** Overstated drift or inconsistent severity across source kinds.
- **Validation:** Null simulations, constant/small/disjoint-support datasets, controlled shifts and statistical power/error-rate analysis. Keep drift outcome (`pass/warn/fail`) separate from execution status.

## P1 — ML evaluation and runtime diagnostics

- **File/functions:** `model_leaderboard.py`, `target_intelligence.py`, `predictive_diagnostics.py`, `analysis/{model,runtime_model,optimizer,architectures}.py`.
- **Problem:** Splits, preprocessing leakage, repeated entities/time ordering, gradient/activation applicability and state preservation need independent review.
- **Reproduction:** Existing runtime safety/PyTorch tests; add grouped/time-indexed prediction datasets and models with BatchNorm/Dropout/shared weights.
- **Expected behavior:** Fit-only preprocessing, appropriate evaluation splits, no target leakage, meaningful degenerate-target behavior, documented temporary model-state changes and exact restoration.
- **Actual implementation:** Bounded baselines, architecture heuristics and optional runtime hooks exist. Platform tests now run with CPU PyTorch but are not a research-quality evaluation audit.
- **Risk:** Optimistic metrics or misleading training advice.
- **Fixtures:** Extend `tests/test_diagnostic_safety.py`, `test_structured_extended.py`, plus new grouped/time leakage golden datasets.
- **Validation:** Compare trusted sklearn pipelines and analytic tiny models; verify model/optimizer state before and after both success and exceptions; seed every stochastic method.

## P1 — Relational inference precision

- **File/functions:** `analysis/relational.py` key/relationship inference and `analysis/structured_contracts.py`.
- **Problem:** Shared names, integer IDs and sample inclusion can suggest false joins.
- **Reproduction/fixture:** `relational_false_join`: unrelated tables with overlapping IDs; add true one-to-many, missing foreign keys and many-to-many duplicates.
- **Expected behavior:** Distinguish inferred candidate from declared relationship; report cardinality and coverage on the correct population.
- **Actual implementation:** Relationships are inferred from bounded table diagnostics; no domain-specific ground-truth calibration.
- **Risk:** Incorrect referential-integrity or join-expansion advice.
- **Validation:** Synthetic relational oracles with known constraints; track false positives/negatives by table size and key skew.

## P2 — Rust sketches, numerical stability and backend agreement

- **File/functions:** `rust/framevitals-core/src/{lib,sketches,fused_profile,categorical_sketches,arrow_scan}.rs`; `rust/framevitals-py/src/lib.rs`, `backends.py`, `streaming_profile.py`.
- **Problem:** Passing engineering tests does not establish quantile/cardinality error guarantees, floating-point stability, merge equivalence or finite-value population consistency.
- **Reproduction:** Existing Rust and native Arrow regression suites; benchmark scripts in `benchmarks/`.
- **Expected behavior:** Documented error bounds, stable merge behavior, dtype/null/NaN/Inf conventions, consistent population definitions across Python/native/streaming backends.
- **Actual implementation:** Rust workspace tests and native bridge tests pass; default wheel remains native-free. No new kernels or formulas were introduced.
- **Risk:** Backend-dependent diagnostics, especially wide/streaming profiles and heavy-tailed values.
- **Fixtures:** Exact small oracles, extreme magnitudes, constant and empty inputs, many shards and adversarial ordering.
- **Validation:** Property-based tests, compensated/high-precision references where justified, seeded accuracy/runtime/RSS sweeps. Do not replace bounded sketches with full-materialization implementations.

## Integration rules for Phase 2

1. Create a separate branch from PR #66's final pinned head, and record that parent SHA.
2. Keep HTTP, CLI, TUI and Python consumers using the same Prism engines.
3. Extend `schema_version="1"` additively; scientifically incompatible meanings need explicit versioning and migration notes.
4. Keep missingness, computation status, sample coverage and accuracy as separate concepts.
5. Add analytic golden fixtures before changing each formula; retain engineering regressions and compare native-free/native-enabled modes.
6. Run full Python tests, Rust checks, frontend build and Chromium workflows after API/schema changes. Compare `benchmarks/platform_probe.py` results in fresh processes.
7. Update this backlog with validated conclusions, remaining uncertainty and any changed thresholds. Never label a newly added method exact merely because it returned successfully.
