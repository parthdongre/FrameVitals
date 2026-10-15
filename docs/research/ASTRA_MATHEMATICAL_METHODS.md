# Astra mathematical methods and interpretation

Scientific methods revision **2**, additive to transport schema `1`. Source baseline: Sol `4487150be3834bafd7b1879832d1a90ec5da2b10`. This revision corrects definitions and identifies approximations; it does not certify all model behavior or infer domain requirements from data alone.

## Spectra, rank and conditioning

For singular values s₁≥…≥sᵣ≥0, numerical rank counts sᵢ > max(m,n) ε₆₄ s₁. The tolerance is exposed in original units. Condition number uses s₁/s_min only when all minimum-dimension singular values exceed that tolerance. A numerically deficient matrix returns null with `condition_status=singular_under_tolerance`, rather than discarding zero singular values and reporting the condition on its nonzero subspace. This intentionally treats nearly singular matrices as numerically singular at the stated tolerance. Null is not a finite condition number.

Stable rank is Σsᵢ²/s₁². Legacy `effective_rank` retains its energy-entropy meaning exp(−Σpᵢ log pᵢ), pᵢ=sᵢ²/Σsⱼ². This differs from the singular-value entropy definition sometimes called effective rank; the explicit definition now travels with the result. Zero matrices use rank and stable rank zero, energy entropy zero by convention, and unavailable condition number.

Matrices are divided by their largest absolute entry before SVD/energy calculations. Rank and ratios are invariant to this scaling; singular values and tolerance are converted back to original units. This prevents squaring 1e200 or 1e−200 entries from corrupting dimensionless summaries. A norm genuinely exceeding float64 remains outside representable output range.

Small matrices use full SVD. Larger matrices use sklearn randomized SVD with 64 components, three power iterations and seed 42. For its orthogonal approximation, relative Frobenius residual is sqrt(max(0,1−Σŝᵢ²/||A||F²)). It is a reconstruction diagnostic of the analyzed matrix, not a probabilistic error bound on its singular values. Stable rank from the approximate spectral norm is an upper estimate. Top-k entropy is entropy within captured energy, not full effective rank.

Above the existing bounded matrix-size policy, rows and columns are selected uniformly without replacement, seed 42. `population=sampled_submatrix` is explicit. Submatrix stable rank is stored separately; full-matrix stable rank and ratio become null. No submatrix-based full-source collapse penalty/finding is emitted. Random row/column sampling removes deterministic stride aliasing but can miss localized structure; it does not yield an unbiased rank estimator. Original and analyzed shapes, truncation and residual population are retained.

Entry distribution sampling also uses seeded uniform positions. NaN/Inf counts scan all NumPy values up to 10 million entries in chunks of 65,536; framework tensors are fully checked up to 1 million entries. Larger sources expose exact counts for the checked sample, `coverage=sampled`, and `all_finite=null`. Absence in a sample is never complete finiteness evidence. Complex distributions are explicitly unsupported rather than silently dropping imaginary parts.

References: [NumPy rank](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html), [NumPy condition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html), [Halko, Martinsson & Tropp](https://arxiv.org/abs/0909.4061).

## Statistics and inference

Finite values define numeric inference populations. Constant samples do not receive a normality conclusion. Non-rejection of normality is not the probability of normality. Shape statistics use bias-corrected Fisher–Pearson skewness and Fisher excess kurtosis, with undefined small/constant cases retained as unavailable.

All distribution candidates are fit and evaluated on the same original observations and units. AIC=2k−2ℓ and BIC=k log(n)−2ℓ. Previous beta-only rescaling/clipping compared incompatible likelihoods. Parameter values now retain float precision so likelihoods can be reproduced. Fitted support parameters and candidate restrictions can violate regular model-selection assumptions; this remains exploratory ranking, not a distribution certificate.

BCa intervals use 999 resamples, seed 42, batch 64. Nonfinite BCa bounds trigger percentile fallback. A constant empirical distribution can yield [c,c]; this describes the resampling distribution, not proof of population certainty. Bootstrap independence/exchangeability assumptions remain essential. The fast Student-t mean interval is exact under IID Gaussian sampling and otherwise asymptotic when finite variance and regularity support it. Its previous absolute small-standard-deviation cutoff is removed. Median intervals retain binomial order-statistic coverage for IID observations; tied/discrete samples can be conservative. Confidence inputs must be in (0,1).

Cramér's V uses uncorrected Pearson χ²: sqrt(χ²/[N min(r−1,c−1)]). The effect size remains descriptive. χ² p-values are withheld when any expected cell count is below five. Numeric pair correlations reject constant inputs. Tiny groups are removed together with their labels. Categorical entropy and its maximum use identical observed support, including the missing category when included in counts.

Bivariate panels add Benjamini–Yekutieli q-values across all returned valid hypotheses. The family is the bounded returned panel, not every possible dataset hypothesis. BY multiplies the BH adjustment by Hₘ=Σ1/i, allowing arbitrary dependence for valid marginal p-values. Existing unadjusted fields remain; adjusted significance is separate. Invalid marginal inference cannot be repaired by adjustment.

References: [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html), [χ² contingency](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html), [FDR control and original references](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## Drift and sequential observations

Numeric drift retains reference-quantile PSI and reference-scale normalized Wasserstein distance. These are effect diagnostics; fixed severity thresholds are heuristic. `z_shift` is a mean shift divided by reference standard deviation, with current-standard-deviation fallback, not a test statistic with a claimed standard-normal null.

Energy distance adds a bin-free descriptive effect: sqrt(2 E|X−Y|−E|X−X′|−E|Y−Y′|). One-dimensional SciPy evaluation sorts samples and uses O((n+m) log(n+m)) work, avoiding pairwise quadratic storage. Its units are square root of input units. No p-value is invented from the effect size.

KS p-values assume independent samples and continuous distributions; ties can make the continuous-null test conservative. Raw p-values retain precision. Drift significance uses BY across returned valid column tests at q<0.01. The overall verdict also includes schema, missingness and heuristic effect thresholds; it has no global calibrated false-alarm guarantee.

Categorical χ² keeps the first 30 categories plus a pooled tail, avoiding silent deletion of tail observations. Expected-count eligibility is checked after pooling. PSI/JS still use the complete categorical support. Very small samples remain explicitly unavailable. Datetime values are converted to nanosecond units explicitly before distance calculation, avoiding pandas storage-unit-dependent magnitudes.

FFT/STL/ACF/PACF/stationarity/forecast calculations require a regular, unique, finite paired time axis. Irregular, duplicate or nonfinite paired axes return unavailable subdiagnostics. No interpolation or aggregation is silently chosen. FFT period selection used for forecasting observes a prefix strictly within the training data. ADF/KPSS disagreement is labelled inconclusive; it does not establish difference- or trend-stationarity. Forecast output is a historical holdout preview, not guaranteed future accuracy.

References: [energy distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html), [KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html), [statsmodels time series](https://www.statsmodels.org/stable/tsa.html).

## Graphs

Topology counts/components are exact for the input graph. Expensive diagnostics retain depth budgets. Paths use BFS for unweighted graphs and Dijkstra for validated nonnegative costs. Missing specified weights have unit cost. Invalid weights trigger explicitly labelled unweighted fallback unless the caller explicitly requests invalid weighted execution, which raises. Negative-cycle/negative-weight shortest-distance inference is not provided. Zero costs are valid for distances but make weighted betweenness unavailable because shortest-walk multiplicity can be ill-defined.

Sources are sampled uniformly without replacement, seed 42. The distance population is reachable ordered nonself pairs from these sources; unreachable pairs are excluded from mean and counted in reachable ratio. When every source is scanned, these are all-pairs summaries. For sampled sources the pooled mean is a ratio estimator, not an unbiased estimate on arbitrary disconnected directed graphs. Maximum observed distance is a lower bound on diameter. Mean/max/counts use all reached distances; p95 uses a uniform priority sample capped at 50,000 distances (seed 1729), with separate quantile provenance. Working memory is O(V+E+50,000), excluding library transient allocations, rather than retaining O(kV) distances.

A CSR Dijkstra portfolio is restricted to simple weighted graphs with ≥20,000 nodes and ≥2V edges, under existing path budgets. Conversion is included in benchmark time. Multigraphs retain NetworkX to preserve minimum-parallel-edge path semantics. Unweighted CSR was rejected after measured slowdown.

Algebraic connectivity is now λ₂(D−A) of the simple undirected unweighted projection without loops, with disconnected graphs exactly zero. Revision 1 used the normalized Laplacian under the same name. Adjacency spectral radius uses the same projection. Dense eigvalsh applies through 64 nodes; sparse eigsh uses seeded start, tolerance 1e−8 and bounded iterations. Failure is unavailable. Cut/core/clustering/community projections are explicit; Louvain remains seeded heuristic modularity optimization. PageRank is a numerical iterative result, not exact arithmetic.

References: [NetworkX algebraic connectivity](https://networkx.org/documentation/stable/reference/generated/networkx.linalg.algebraicconnectivity.algebraic_connectivity.html), [SciPy Dijkstra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html).

## Streaming and sketches

Python and Rust retain central-moment merge recurrences through M4. They are mathematically exact identities with floating-point rounding, not exact real arithmetic. NumPy normalizes before computing moments to keep finite means and dimensionless shape usable at extreme scales; overflowing raw moments are marked `raw_moment_overflow`. Such states are not safe exact merge summaries at that scale. Rust initializes its first finite value directly to prevent inf×0 from making the moments of a huge constant NaN. Unrepresentable Rust derived moments return unavailable. There is no new arbitrary-precision accumulator claim.

For logarithmic bins [γᵏ,γᵏ⁺¹), γ=(1+α)/(1−α), use representative (1+α)γᵏ. Its relative error is bounded by α against either bin endpoint. The former geometric midpoint γᵏ⁺½ exceeds α at the lower endpoint. Both Python and Rust use the corrected representative. Merges require exactly identical accuracy configuration; Python zero thresholds must also match. Quantiles use the lower order statistic floor(q(N−1)), not NumPy's default linear interpolation. Values within the zero threshold have absolute error bounded by that threshold; relative error is not claimed there. Log/exp floating-point rounding adds numerical error near extremes.

HyperLogLog, Misra–Gries-style heavy hitters and priority reservoirs retain existing implementations; distribution-level estimator guarantees are not independently established by this revision. HLL estimates are probabilistic, heavy-hitter counts are summaries rather than exact frequencies, and partition IDs/sequence identity matter for deterministic reservoir reproducibility.

Reference: [Masson, Rim & Lee, DDSketch](https://arxiv.org/abs/1908.10693). FrameVitals' sketch is related; it is not asserted to implement every DDSketch feature or collapsing guarantee.

## Models, anomalies, relationships and health

`run_model_leaderboard` adds explicit `split_strategy=iid|group|time`, `group_column`, `time_column`, and `time_gap`. Split columns are excluded from predictors. Group folds keep entities disjoint; time folds follow sorted unique timestamps, with a gap. IID defaults remain. Imputation/scaling/encoding are fit inside each fold. Whole-dataset conservative feature eligibility remains a disclosed transductive limitation. Fold standard deviations are descriptive, not confidence intervals. Only models with every primary fold valid are ranked. Balanced accuracy and macro F1 supplement weighted metrics. Selected-model refits on previously CV-exposed data are explicitly non-independent. Group/time evaluation does not manufacture a random holdout. Nested evaluation and group/time routing through higher-level convenience protocols remain future work.

Anomaly scores are in-sample relative scores, not probabilities or guaranteed contamination rates. All-zero detector scores no longer cast positive anomaly votes. Positive ties may still exceed nominal contamination. Robust covariance fallback consistently uses squared Mahalanobis distance. Synthetic benchmark precision does not prove universal superiority of an ensemble. Neural reconstruction remains an optional bounded in-sample diagnostic.

Model/runtime samples preserve float64 instead of narrowing all tensors to float32. Coverage and seed accompany sampled observations. Runtime observation is a single eval-mode batch with temporary hooks and restored module flags/RNG; arbitrary user forward code may mutate custom state. Static projection/gate norms cannot certify training dynamics, robustness or generalization. Architecture thresholds remain heuristic; no calibrated failure probabilities are claimed.

Relational equality no longer string-coerces distinct key types. Coverage is labelled unique nonmissing-key coverage, with separate row-weighted coverage. Relationships remain inferred candidates. Prefix sampling can bias uniqueness and join conclusions. Nested traversal distinguishes shared object aliases from ancestor cycles; wildcard path conflicts remain observational structure diagnostics.

All domain health formulas remain legacy heuristics; no replacement weights are introduced without calibration data. The envelope exposes uncalibrated/non-probabilistic/non-comparable status and lists critical findings that must override score interpretation. Tabular health exposes its exact component weights and offset. A high score can coexist with a critical defect, and low rank/sparsity/disconnection may be intentional.

Reference: [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
