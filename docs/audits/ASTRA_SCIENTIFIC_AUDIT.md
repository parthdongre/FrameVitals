# Astra scientific audit

## Scope and verified base

Phase 2 starts at Sol `4487150be3834bafd7b1879832d1a90ec5da2b10`, not main. PR #66 was initially open, then merged into `feature/structured-data-graph-engine` as `b06684e`; that merge has the identical tree. The science PR therefore targets the surviving feature branch. Sol's deleted working branch is not recreated or rewritten.

The original expanded baseline passed 574 Python tests. Before numerical edits, an independent analytic suite produced **18 failures / 4 passes**. Those results reproduced definite defects and exposed missing provenance; failures involving new metadata do not individually prove a numerical formula wrong. After an idle workspace reset the code/tests were reconstructed from the recorded evidence and checkpointed on the science branch. All final results below are from fresh runs, not inferred from the earlier environment.

Verification labels are scoped: **Verified** means a stated deterministic property was checked against an independent oracle; **Verified with assumptions** requires the assumptions named below; **Approximate with known error characteristics** does not mean every dataset has a quantified uncertainty interval. **Mathematically questionable** and **Unsupported or unverified** remain explicit outcomes. This audit is not a blanket scientific certification.

## Confirmed defects and corrections

| Implementation | Classification before → after | Evidence and correction |
| --- | --- | --- |
| tensor condition number | Incorrect → Verified with assumptions | diag(3,0) returned 1 after zero singular values were removed. Now null/singular status under the exposed rank tolerance. Diagonal oracles at 1e±200 pass. |
| tensor spectral energy | Incorrect → Verified with assumptions | Unscaled squaring overflowed/underflowed dimensionless ranks. Max-entry scaling restores scale invariance within representable outputs. |
| strided spectral sampling | Mathematically questionable → Approximate with known error characteristics | A periodic diagonal aliased under stride. Seeded uniform submatrix sampling identifies its own population; no full-rank estimate or penalty is manufactured. |
| tensor nonfinite checks | Unsupported coverage → Verified checked counts | Isolated omitted-tail NaN fixture now detected by bounded full scan. Larger inputs explicitly retain sampled coverage and unknown full finiteness. |
| complex distributions | Incorrect → Unsupported explicitly | Casting to real discarded imaginary components. Such distributions are now explicitly unavailable. |
| distribution AIC | Incorrect → Verified likelihood consistency; assumptions remain | Beta-only rescaling/clipping compared likelihoods in different units. All candidates now use original observations. Independent logpdf recomputations match. |
| BCa availability | Incorrect → Verified bounds handling | Constant data returned available=true with null bounds. Nonfinite BCa triggers percentile fallback. |
| Cramér's V | Incorrect → Verified descriptive definition | Continuity-corrected χ² reduced perfect binary association below one. Effect size uses uncorrected Pearson χ²; sparse-cell inference is unavailable. |
| entropy normalization | Incorrect → Verified | Missing category appeared in entropy but not maximum entropy. Numerator/denominator now use identical support. |
| group-test labels | Incorrect → Verified | Removing undersized samples failed to remove their labels. Samples/labels now stay aligned. |
| tiny confidence intervals | Incorrect → Verified with assumptions | Absolute cutoff and output rounding could collapse nonzero tiny intervals. Precise endpoints retained; unsupported near-unit median confidence is unavailable. |
| multiple testing | Mathematically questionable → Verified adjustment with assumptions | Raw p-values alone inflated interpretive risk. BY matches SciPy's independent reference, including correlated-test policy; effect thresholds remain heuristic. |
| categorical drift tail | Incorrect inferential population → Verified observed table | χ² discarded categories beyond 30. Pooled tail retains all observations; expected counts checked. |
| datetime drift units | Incorrect when storage units differ → Verified conversion | Integer datetime representation is explicitly normalized to nanoseconds before days conversion. |
| algebraic connectivity | Incorrect name/definition → Verified with assumptions | Revision 1 returned normalized Laplacian λ₂ under an unqualified name. Revision 2 returns combinatorial λ₂; P4 analytical oracle passes. Projection/solver are explicit. |
| graph path provenance | Unsupported aggregate sampling → Approximate with known error characteristics | Seeded uniform source sampling names reachable ordered-pair population. Exact mean/max and bounded sampled quantiles are distinct. |
| graph path execution | Verified with assumptions | CSR weighted portfolio differentially matches NetworkX; budgets, negative-weight handling and multigraph fallback retained. Unweighted CSR rejected by measurement. |
| quantile representative | Incorrect declared bound → Verified with floating-point assumptions | Geometric midpoint can exceed α at a lower bin endpoint. Representative (1+α)γᵏ obeys endpoint bound. Python and Rust regressions cover α=.01,.1,.5, signs and bins. |
| sketch merge configuration | Incorrect → Verified | Python's isclose accepted incompatible bin/zero configurations. Exact configuration equality required. |
| streaming extreme constant | Incorrect → Verified bounded stress case | Rust first observation could evaluate inf×0; direct initialization fixes huge constants. NumPy normalization protects dimensionless shape, flags raw overflow. |
| ML ranking incomplete folds | Mathematically questionable → Verified policy | Partial successful folds no longer compete against complete results. Every primary fold must be finite. |
| ML split dependence | Unsupported beyond IID → Verified split invariants | Explicit group/time split options prevent group overlap/future-to-past train leakage; preprocessing stays inside folds. Whole-data eligibility screen disclosed. |
| selected-model holdout | Misleading independence → Verified interpretation | Refit uses observations already used by CV selection; now explicitly non-independent. No nested-CV claim. |
| anomaly zero votes | Incorrect agreement → Verified policy | All-zero detector scores met a zero quantile and voted every row. Positive-score requirement prevents this. Positive ties remain disclosed. |
| Mahalanobis fallback | Inconsistent score transform → Verified definition | Robust path returned squared distances while fallback square-rooted them before ensemble normalization. Both now use squared distances. |
| model sample precision | Incorrect for float64 extremes → Verified sampling policy | Unconditional float32 conversion could create Inf or zero. Shared sampler preserves float64 and reports seed/coverage. |
| runtime interpretation | Mathematically questionable → Verified observation limits | Single-batch magnitude outliers no longer carry definitive exploding/vanishing titles. Existing state/RNG/hook preservation is independently checked after an exception. |
| relational equality | Incorrect → Verified observed equality | Integer 1 and string '1' became equal after str conversion. Native key types retained. Unique-key and row-weighted coverage are separate. |
| nested alias cycles | Incorrect → Verified | Shared acyclic objects were called cycles. Ancestor references and shared aliases are counted separately. |
| irregular time series | Incorrect preconditions → Verified gating | FFT/lag tests ran after deleting irregular gaps. Unique regular finite paired axes now required; ADF/KPSS disagreements are inconclusive. |

Golden evidence: `tests/test_scientific_golden.py` and `tests/fixtures/science/astra_golden.json`; legacy tests remain. One old large-tensor assertion was migrated because it required the scientifically invalid full-matrix inference from a submatrix. The replacement asserts the submatrix diagnostic and the absence of that unsupported full-source finding; its numerical threshold was not weakened.

## Coverage of analytical families and remaining uncertainty

The following inventory distinguishes code inspection, deterministic tests and statistical validation. Formulas, domains and detailed tolerances are in [methods](../research/ASTRA_MATHEMATICAL_METHODS.md). No passing integration test is promoted to a distribution-wide guarantee.

| Modules / algorithm family | Final classification | Preconditions, cost and failure limits / remaining evidence |
| --- | --- | --- |
| profiler, backends, analysis_state: mean/M2–M4 | Verified with assumptions | Finite real values; O(N) scan, O(1) merge/state per column. Five seeded partition tests and six native differentials; high-offset cancellation and unrepresentable raw moments still limit precision. |
| streaming_profile, streaming_exact_reuse, fused_profile, arrow_scan | Verified with assumptions for existing dtype/population regressions | O(cells) scan with schema/batch budgets. Existing Arrow/native tests cover dtype/null/Inf routing. No universal bitwise backend equivalence. |
| Python/Rust log quantiles | Approximate with known error characteristics | Relative error against lower order statistic away from zero; exact merge configuration; O(N) bin insertion, query O(B log B), storage O(B). Extreme exp/log rounding remains. |
| HLL, categorical sketches | Unsupported or unverified at distribution-wide guarantee level | Existing deterministic cardinality/merge tests pass; probabilistic RMS error curves and adversarial hash workloads need a larger study. |
| heavy hitters, priority reservoir | Verified bounded-state invariants; estimator accuracy unverified | Existing merge/boundedness tests. Priority hash collision, stream identity and heavy-hitter residual-count interpretation require further audit. |
| deep_statistics, deep_statistics_v2, fast_deep_statistics | Verified with assumptions for corrected methods | O(N) descriptions, pair budget; SVD/fit/bootstrap variable work. Parametric assumptions, fitted-support AIC regularity and bootstrap coverage require domain validation. Legacy normaltest/χ² paths still lack the full v2 policy. |
| correlations, target_intelligence, target_leakage | Verified with assumptions for association definitions; leakage status heuristic | Correlation/purity are observational; target-derived features can also be legitimate predictors. No causal or temporal-leakage completeness guarantee. Pair/cardinality bounds can omit signals. |
| multicollinearity | Mathematically questionable | VIF uses median imputation, caps near-collinear output at 999, groups around a pivot rather than equivalence classes. Treat 999 as legacy saturation, not exact VIF. Replacement/deprecation needs coordinated UI work. |
| health_score and every domain score | Mathematically questionable as risk / verified deterministic heuristic | Arbitrary domain penalties, not calibrated probabilities or comparable modalities. Component weights/offset and critical override interpretation exposed; calibration data absent. |
| fast_anomaly, anomaly_ensemble | Approximate heuristic, limited controlled evidence | O(NDP) projections vs tree/neighborhood/covariance fitting; model and seed sensitive. Shifted-Gaussian benchmark detects planted tails; no universal ensemble improvement. Missingness/imputation/high dimensionality need more workloads. |
| neural_anomaly | Unsupported or unverified as anomaly generalization | Bounded MLP in-sample reconstruction; convergence warning visible. Training contamination and fitting/scoring same data prevent calibrated novelty claims. |
| model_leaderboard, baseline_model, predictive_diagnostics | Verified split/fold mechanics where tested; rankings conditional | CV training cost times models/folds; default IID. Group/time options are direct leaderboard API only. Baseline/predictive convenience APIs retain IID assumptions. No nested selection uncertainty or winner significance test. |
| ml_preprocessing, feature_importance, explainability | Verified fit isolation in existing tests; heuristic explanatory scope | Conservative eligibility uses whole dataset. Importance/SHAP are predictive, not causal; correlated predictors and leakage can distort interpretation. No new heavy dependency. |
| analysis/model, runtime_model, optimizer, architectures | Verified structural/state invariants; threshold claims heuristic | Bounded samples/modules; some architecture scans still stride/cast and can miss defects. Runtime uses eval semantics and cannot guarantee arbitrary custom-forward purity. Static gate spectral radii do not establish nonlinear recurrent stability. |
| safetensors, onnx_model | Verified metadata extraction in engineering tests; scientific accuracy unverified | Shapes/byte counts/operators are observations; no inference accuracy or external-weight inspection. Safetensors comparison is metadata-level, not a weight-level equality certificate. ONNX nested control-flow semantics need deeper validation. |
| graph BFS/Dijkstra/components/cuts/core | Verified with assumptions / projected topology explicit | Analytic path/cycle/complete cases, weighted differential benchmark; O(V+E) BFS/components, O((V+E)log V) binary-heap Dijkstra per source. Projection can change directed/multigraph interpretation. |
| PageRank/betweenness/Louvain/clustering | Approximate with known implementation characteristics; broader estimator calibration unverified | PageRank iterative tolerance, seeded betweenness sources, heuristic communities, simple undirected clustering projection. No community truth or significance claim. |
| graph eigenanalysis | Verified definitions on small graphs; numerical approximation for sparse solver | Dense O(V³), sparse iteration depends on spectral gap. Seed/tolerance/iteration cap explicit; convergence failures unavailable. |
| analysis/change graph/tensor/model drift | Mathematically questionable as calibrated change severity | Node/count/degree summaries miss rewiring with identical degree sequence. Tensor aligned norms still have small-scale/overflow sensitivity; file checkpoint comparison has limited scope. Higher-order identity/change tests remain work. |
| nested, structured_contracts | Verified cycle/alias/type fixtures; heuristic inferred constraints | Bounded traversal and wildcard types; sampling/order and alias deduplication influence counts. Truncation is explicit, no complete-schema guarantee. |
| relational | Verified observed coverage/join arithmetic; inferred constraints unverified | Unique-key and row populations separate. Prefix sample uniqueness is not population uniqueness; shared names do not establish keys. Duplicate ID findings can be valid many-side structure. |
| document | Verified extracted-text counts under parser assumptions; substantive quality unverified | Text extraction/truncation/no OCR limit semantic conclusions. Health is extraction heuristic. Hash comparison covers extracted portion only. |
| time_series | Verified regularity/diagnostic preconditions; inferential calibration conditional | FFT O(N log N); STL/ADF/forecast costs vary. Automatic period/series selection and multiple tests are exploratory, not a forecasting certification. |
| stream_change | Approximate heuristic / calibration unverified | Clipped scale-adaptive CUSUM on batch means with decay, not a standard Page–Hinkley guarantee. Threshold false-alarm/delay depends on windowing and process. |
| quality_diagnostics, advanced_indicators, ml_readiness, column_roles, semantic_types | Verified engineering fixtures; heuristic scientific judgments | Threshold/name-based roles, IDs, missingness relationships, fairness prompts and utility are observational rules. No fairness guarantee, probability calibration or complete issue detector. |

## Interpretation and compatibility

Transport schema remains `1`; additive `scientific_methods_version=2` and per-method metadata identify changed scientific meanings. `tensor.matrix.method_version=2` and `graph.spectral.method_version=2` identify incompatible interpretations. Full-source stable-rank fields are null for submatrices, with replacement `submatrix_*` fields. Rank-tolerance deficiency produces null condition number plus explicit status. Graph algebraic connectivity switches to the combinatorial definition. Sketch representative values change within the newly justified bound. Drift significant booleans now use q<.01, preserving unadjusted flags and p-values. The browser caption names full-matrix versus submatrix/randomized spectra.

No new default runtime dependency or GPU dependency is introduced. All baseline public entry points remain. Missing/invalid computations remain distinguishable from zero. New methodology cannot transform a heuristic confidence number into an estimated probability.

## Validation and research

See [performance report](../benchmarks/ASTRA_PERFORMANCE_REPORT.md), [algorithm selection](../research/ASTRA_ALGORITHM_EVALUATION.md) and [methods](../research/ASTRA_MATHEMATICAL_METHODS.md). References there are primary papers and official library documentation. Integration, package, native-free and native-enabled results are reported separately; unsupported claims in the table remain open even when every automated test passes.
