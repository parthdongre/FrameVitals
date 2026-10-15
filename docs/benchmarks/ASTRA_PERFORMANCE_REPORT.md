# Astra performance and validation report

## Reproducible comparison

Baseline: Sol `4487150be3834bafd7b1879832d1a90ec5da2b10`. Same Python 3.12 environment and installed numerical libraries, fresh subprocess per workload/seed, seeds 11/29/47, BLAS/OMP/MKL threads fixed at one. Three runs per implementation. Timings exclude imports/input generation except lazy algorithm imports and CSR conversion inside the measured function. Peak RSS includes process/import/input memory; Python allocation peak is measured in a separate second call, not during timed execution. This is a small seeded benchmark, not a controlled cross-machine speed guarantee. See raw [JSON](astra-results.json) and `benchmarks/astra_science.py`.

| Workload | Before median (s) | After median (s) | Before peak RSS (MiB) | After peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| graph | 0.05545 | 0.06466 | 131.7 | 118.3 |
| graph_weighted | 1.11505 | 0.56886 | 181.0 | 197.3 |
| tensor | 0.64274 | 0.82917 | 188.1 | 206.6 |
| sketch | 0.00742 | 0.01003 | 101.6 | 101.8 |
| moments | 0.00151 | 0.00170 | 155.3 | 156.2 |
| drift | 0.06692 | 0.08136 | 161.4 | 161.1 |
| anomaly | 0.01870 | 0.01631 | 102.8 | 102.8 |

## Accuracy, cost and decisions

- Unweighted path: 10,000 nodes, 12 sources. BFS retained. New bounded priority quantile sampling trades some work for bounded storage; mean error versus the analytical reachable-pair oracle stays below 5e−7 (public rounding). This final run is slower than the old unbounded implementation, while using less peak process memory. No unweighted speedup claim.
- Weighted sparse directed graph: 20,000 nodes, probability .0002, roughly 80,000 edges, positive uniform costs, 12 sources. New CSR portfolio is about 1.96× faster in this run, including conversion; peak RSS increases about 16 MiB. Different source selection schemes each use an independent NetworkX oracle over their actual sources; mean error is below 5e−7. Accuracy comparisons do not compare different sampled populations as if identical.
- Tensor: 1800×640 Gaussian matrix, bounded distribution sample and 64-component randomized spectrum. Both analyze the full matrix for this shape. New entry sampling, bounded full nonfinite scan and metadata add overhead. Full-source stable-rank values agree to public baseline precision. This benchmark is not a submatrix accuracy guarantee; the aliasing golden fixture tests the submatrix policy separately.
- Lognormal quantile sketch: 100,000 values, log-scale standard deviation four. Maximum relative error across five queried lower order statistics remains below .01 for all three runs. Boundary goldens establish the corrected bin error; this small sample alone would not establish the bound. The corrected implementation is not uniformly faster.
- Streaming moments: 100,000 Student-t(6) values. Variance, skewness and kurtosis agree with independent NumPy/SciPy within floating-point tolerances. Normalization costs extra work and improves extreme-scale robustness. Raw fourth moments can still be unrepresentable.
- Drift: 2000×24 independent Gaussian reference/current tables, one shifted column. Shift is detected severe; zero null columns crossed the reported significance threshold in these particular seeds. BY adjustment and energy-distance computation cost extra time. These three seeds do not estimate a reliable global false-alarm rate; heuristic severity can warn under a valid null.
- Anomalies: 1000×8 Gaussian data with the final 50 rows shifted by six. Precision@50 is 1.0 in these easy controlled cases. This does not establish ensemble superiority, realistic false-positive rate or robust high-dimensional performance. Timing variation is not claimed as a kernel optimization.

No new runtime or GPU dependency is introduced. The largest measured gain is workload-specific weighted graph execution. Some scientifically necessary checks increase time/memory; the report retains those regressions. Large graph/tensor tests remain bounded, but the report does not claim out-of-core linear algebra, arbitrary parser isolation or a proven optimal dispatch threshold.

## Validation observed on the final implementation

| Check | Result |
| --- | --- |
| Expanded native-enabled Python | 627 passed, no skips; five visible warnings |
| Independent scientific golden cases | 53 passed as part of full suite |
| Separate pure-wheel core-only environment | 510 passed, 68 expected optional skips; one visible warning |
| Rust workspace | 22 core tests passed; bridge/doc targets passed |
| Rust formatting / Clippy all targets with warnings denied | Passed |
| Maturin native bridge build | Passed |
| Chromium real API workflows | 4 passed, including scoped axe scans |
| Frontend TypeScript/Vite build | Passed |
| Pure wheel/sdist and Twine | Passed; pure wheel contains no .so/.pyd |
| Critical/unused Python lint and whitespace | Passed |

Warnings are degenerate graph assortativity and deliberately bounded neural optimizer convergence, not suppressed failures. Core-only skips are optional parsers, integrations, native/PyTorch and graph tests, not disabled mathematical assertions. macOS/Windows live terminals, Safari, GPU runtime, distributed merges, estimator confidence coverage and recent-paper SSSP reproduction were not executed.

Native linking in this runtime required a local scratch symlink to the existing libpython3.12 shared library and an RUSTFLAGS search path because sysconfig pointed at /install/lib. No environment-specific workaround was committed. Browser installation initially failed during package-manager privilege changes; downloading the browser alone succeeded and all workflows ran.

## Repeat commands

```bash
git worktree add --detach ../FrameVitals-sol 4487150
python benchmarks/astra_science.py --before ../FrameVitals-sol --after . --output docs/benchmarks/astra-results.json
python -m pytest -q
python -m ruff check src/framevitals tests benchmarks/astra_science.py --select E9,F401,F821,F822,F823,F841
python -m build
python -m twine check dist/*
cd rust
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cd ../frontend
npm ci
npm run build
npx playwright install chromium
npm test
```

Validation pins and live CI status are recorded in the PR. Tested implementation commit: `b16cfe94e832f8ac9b6bbbfb09404386023dec62`, tree `9e9f488dc48034b8697a17f3072d9904933a1ba3`. Documentation-only publication commits are distinct from the tested implementation; no release, merge or deployment is performed by this work.
