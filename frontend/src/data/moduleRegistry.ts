/**
 * Module registry — single source of truth for
 *   - the `/modules` page listing
 *   - the per-tab `<HowThisWorks/>` disclosures
 *
 * Each entry maps to one or more report tabs by `id`. The id is the same as
 * the tab id used in `pages/report/tabRegistry.ts`, with a few extra entries
 * (`profile`, `analysis_selector`, etc.) that aren't tabs but appear on the
 * `/modules` page.
 */

export interface ModuleRegistryEntry {
  id: string;
  name: string;
  category:
    | "Profiling"
    | "Statistics"
    | "Quality"
    | "ML"
    | "Time"
    | "Text"
    | "Drift"
    | "AI"
    | "Cleaning";
  /**
   * One-line description for the modules table.
   */
  oneLiner: string;
  /**
   * What the module produces.
   */
  what: string;
  /**
   * How the module works algorithmically.
   */
  how: string;
  /**
   * Algorithms/libraries actually used.
   */
  algorithms: string[];
  /**
   * Source path under the repo root.
   */
  source: string;
}

export const MODULES: ModuleRegistryEntry[] = [
  /* ---- Profiling ----------------------------------------------------- */
  {
    id: "profile",
    name: "Dataset Profile",
    category: "Profiling",
    oneLiner: "Schema, dtypes, missingness, duplicates, correlations.",
    what:
      "Computes shape, dtypes, missing counts, duplicate rows, numeric/categorical/date column lists, descriptive statistics for numeric columns, top values for categorical columns, a Pearson correlation matrix, and a 15-row preview.",
    how: "Pandas describe + dtype inference + per-column null rates + Pearson/Spearman correlation matrix.",
    algorithms: ["pandas", "numpy"],
    source: "modules/loader.py + modules/profiler.py",
  },
  {
    id: "rolesSummary",
    name: "Column Roles",
    category: "Profiling",
    oneLiner: "Tags each column with semantic roles.",
    what:
      "Assigns id_like, sensitive, target_candidate, high_cardinality, constant, severe_missing, and numeric_meaningful tags so downstream modules can make smarter decisions.",
    how: "Combines column-name keyword matching, cardinality ratios, missing percentages, and dtype checks into a small rule engine.",
    algorithms: ["keyword matching", "cardinality + missingness rules"],
    source: "modules/column_roles.py",
  },
  {
    id: "datasetSignals",
    name: "Dataset Signals",
    category: "Profiling",
    oneLiner: "Boolean flags about the dataset as a whole.",
    what:
      "has_potential_leakage, has_class_imbalance, has_high_cardinality, has_dates, has_text, has_severe_missing, and similar flags.",
    how: "Aggregates the profiler + column-roles output through small rule predicates.",
    algorithms: ["rule predicates"],
    source: "modules/dataset_signals.py",
  },
  {
    id: "analysis_selector",
    name: "Analysis Selector",
    category: "Profiling",
    oneLiner: "Decides which analyses to actually run.",
    what:
      "Given the dataset signals + analysis mode, marks each analysis as selected, recommended, or skipped.",
    how: "Looks up an analysis inventory and applies signal-driven gates per analysis.",
    algorithms: ["signal-gated selection"],
    source: "modules/analysis_selector.py",
  },

  /* ---- Statistics ---------------------------------------------------- */
  {
    id: "statistics",
    name: "Deep Statistics v2",
    category: "Statistics",
    oneLiner: "Univariate + bivariate statistics with effect sizes.",
    what:
      "Per-column normality battery (Shapiro · D'Agostino · Anderson), best-fit distribution by AIC across six candidates, BCa bootstrap CIs, and bivariate tests including Mann-Whitney, Kruskal-Wallis, Cramér's V, and point-biserial.",
    how: "scipy.stats normality tests, AIC distribution selection, BCa bootstrap, bivariate effect-size estimators.",
    algorithms: ["scipy.stats", "numpy"],
    source: "modules/deep_statistics_v2.py",
  },

  /* ---- Anomalies ----------------------------------------------------- */
  {
    id: "anomalies",
    name: "Anomaly Ensemble",
    category: "Quality",
    oneLiner: "Multi-detector outlier scoring.",
    what:
      "Seven detectors run in parallel: IsolationForest, LOF, EllipticEnvelope, robust z-score, Mahalanobis, ECOD, COPOD. Their normalized scores are averaged into a single ensemble.",
    how: "Each detector emits a [0,1] score, scores are averaged, and rows above the configured threshold are flagged.",
    algorithms: ["scikit-learn", "pyod"],
    source: "modules/anomaly_ensemble.py",
  },

  /* ---- ML lab -------------------------------------------------------- */
  {
    id: "ml-lab",
    name: "Model Leaderboard",
    category: "ML",
    oneLiner: "Quick model bake-off with cross-validated metrics.",
    what:
      "5-fold cross-validated leaderboard across up to 8 models including XGBoost and LightGBM, with calibrated holdout metrics. The winner is selected by primary metric and is the model used for SHAP attribution.",
    how: "Stratified K-fold CV for classification, plain K-fold for regression. Each model is wrapped in a small preprocessing pipeline (impute, encode, scale where appropriate).",
    algorithms: ["scikit-learn", "xgboost", "lightgbm"],
    source: "modules/model_leaderboard.py",
  },
  {
    id: "targetAnalysis",
    name: "Target Analysis",
    category: "ML",
    oneLiner: "Inspects the chosen target column.",
    what:
      "Detects task type (classification / regression), distribution skew, class imbalance, and target leakage candidates.",
    how: "Heuristics on dtype, cardinality, and value distribution; cross-checks against column roles.",
    algorithms: ["pandas", "scipy.stats"],
    source: "modules/target_analyzer.py",
  },
  {
    id: "featureImportance",
    name: "Feature Importance",
    category: "ML",
    oneLiner: "Global feature importance.",
    what:
      "Ranks features by mean absolute SHAP for tree models, falling back to permutation importance.",
    how: "TreeSHAP via the `shap` library; permutation importance from scikit-learn.",
    algorithms: ["shap", "scikit-learn"],
    source: "modules/feature_importance.py",
  },
  {
    id: "baselineModel",
    name: "Baseline Model",
    category: "ML",
    oneLiner: "Reference model the leaderboard compares against.",
    what:
      "A simple baseline (DummyClassifier / DummyRegressor) so improvements over uninformed predictions are visible.",
    how: "5-fold CV with the strategy `most_frequent` (classification) or `mean` (regression).",
    algorithms: ["scikit-learn"],
    source: "modules/baseline_model.py",
  },

  /* ---- SHAP / Explainability ---------------------------------------- */
  {
    id: "shap",
    name: "Explainability (SHAP)",
    category: "ML",
    oneLiner: "Global + per-row attributions for the winner.",
    what:
      "Mean |SHAP| values across the validation set, plus per-row stories that decompose each prediction into its top contributing features.",
    how: "TreeSHAP for tree models, permutation importance fallback for non-tree models.",
    algorithms: ["shap", "scikit-learn"],
    source: "modules/explainability.py",
  },

  /* ---- Time series --------------------------------------------------- */
  {
    id: "timeseries",
    name: "Time Series",
    category: "Time",
    oneLiner: "Frequency, stationarity, decomposition, ACF/PACF, forecast.",
    what:
      "Detects the date column, infers frequency and seasonality, runs ADF + KPSS, performs STL decomposition, computes ACF/PACF, and produces a Holt-Winters forecast.",
    how: "statsmodels for ADF/KPSS/STL/ACF/PACF; Holt-Winters from statsmodels.tsa.holtwinters.",
    algorithms: ["statsmodels"],
    source: "modules/time_series.py",
  },

  /* ---- Text ---------------------------------------------------------- */
  {
    id: "text",
    name: "Text Profile",
    category: "Text",
    oneLiner: "Vocabulary, n-grams, topic scatter.",
    what:
      "Per-column linguistic stats: vocabulary size, average length, regex pattern hits, top n-grams, and a TF-IDF + LSA document map.",
    how: "TF-IDF vectorization, truncated SVD for the LSA scatter, English stopwords filtered.",
    algorithms: ["scikit-learn"],
    source: "modules/text_profile.py",
  },

  /* ---- Drift --------------------------------------------------------- */
  {
    id: "drift",
    name: "Drift",
    category: "Drift",
    oneLiner: "Distribution shift between two datasets.",
    what:
      "Per-column drift report comparing a reference dataset to a current dataset (or one dataset split chronologically).",
    how: "Per-column KS test for numeric, chi-square for categorical, and Population Stability Index. Severity bucketed into stable / minor / moderate / severe.",
    algorithms: ["scipy.stats", "numpy"],
    source: "modules/drift_analysis.py",
  },

  /* ---- Diagnostics --------------------------------------------------- */
  {
    id: "diagnostics",
    name: "Model Diagnostics",
    category: "ML",
    oneLiner: "Residuals, calibration, fairness slices.",
    what:
      "Holdout residuals, reliability bins, and slice-level error metrics for the winning model.",
    how: "Residuals from a holdout split + calibration via reliability diagrams.",
    algorithms: ["scikit-learn", "numpy"],
    source: "modules/model_diagnostics.py",
  },
  {
    id: "multicollinearity",
    name: "Multicollinearity",
    category: "ML",
    oneLiner: "VIF per feature.",
    what:
      "Variance Inflation Factor for each numeric feature; flags variables that are heavily explained by their peers.",
    how: "OLS regression of each feature on the rest, VIF = 1 / (1 - R^2).",
    algorithms: ["statsmodels"],
    source: "modules/multicollinearity.py",
  },
  {
    id: "targetLeakage",
    name: "Target Leakage",
    category: "ML",
    oneLiner: "Feature → target mutual information heuristics.",
    what:
      "Surfaces features whose mutual information with the target is suspiciously high or whose names match common leakage patterns.",
    how: "scikit-learn mutual_info_* + correlation thresholding + name heuristics.",
    algorithms: ["scikit-learn"],
    source: "modules/target_leakage.py",
  },

  /* ---- Segments ------------------------------------------------------ */
  {
    id: "segments",
    name: "Segments",
    category: "ML",
    oneLiner: "Subgroup metrics.",
    what:
      "Group-by slices on detected categorical or binned numeric columns; compares target-relevant metrics per slice.",
    how: "Pandas group-by with a small picker that prefers low-cardinality interpretable columns.",
    algorithms: ["pandas"],
    source: "modules/segment_analysis.py",
  },

  /* ---- Cleaning ------------------------------------------------------ */
  {
    id: "cleaning",
    name: "Cleaning",
    category: "Cleaning",
    oneLiner: "Imputation + dedupe + normalization.",
    what:
      "Removes duplicate rows, imputes missing values (median for numeric, mode for categorical), clips extreme outliers, and writes a cleaned CSV to disk.",
    how: "pandas + scikit-learn pipelines. Health is rescored on the cleaned dataset to quantify the gain.",
    algorithms: ["pandas", "scikit-learn"],
    source: "modules/cleaner.py",
  },

  /* ---- AI ------------------------------------------------------------ */
  {
    id: "ai-report",
    name: "AI Narrative",
    category: "AI",
    oneLiner: "Generated executive summary.",
    what:
      "A grounded narrative report describing the dataset's quality, ML readiness, anomalies, time-series structure, and recommended next steps.",
    how: "LLM agent loop on Ollama (qwen3:4b) with Pydantic-validated output, falling back to OpenRouter, then to a deterministic writer that quotes RAG-retrieved facts.",
    algorithms: ["LLM", "RAG"],
    source: "modules/ai_insights.py + modules/ai_agent.py",
  },
  {
    id: "ask",
    name: "Ask Anything",
    category: "AI",
    oneLiner: "Agentic Q&A grounded in this dataset.",
    what:
      "A chat panel that answers questions about the current dataset using the cached analysis result.",
    how: "Planner → Executor → Critic → Writer loop. Tools are scoped to the cached profile, signals, leaderboard, and explainability — answers cite the exact rows/columns they used.",
    algorithms: ["LLM agent loop"],
    source: "modules/ai_agent.py + /api/ask",
  },
];

export function moduleById(id: string): ModuleRegistryEntry | undefined {
  return MODULES.find((m) => m.id === id);
}
