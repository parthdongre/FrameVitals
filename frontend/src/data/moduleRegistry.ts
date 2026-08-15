/**
 * Canonical analysis-module registry used by the dashboard.
 *
 * Source paths deliberately point at `src/framevitals/`, which is the only
 * maintained Python implementation namespace.
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
  oneLiner: string;
  what: string;
  how: string;
  algorithms: string[];
  source: string;
}

export const MODULES: ModuleRegistryEntry[] = [
  {
    id: "profile",
    name: "Dataset Profile",
    category: "Profiling",
    oneLiner: "Schema, dtypes, missingness, duplicates, and correlations.",
    what: "Builds the structural snapshot consumed by downstream diagnostics.",
    how: "Uses pandas dtype inference, descriptive statistics, null rates, value counts, date detection, and correlation matrices.",
    algorithms: ["pandas", "numpy"],
    source: "src/framevitals/loader.py + src/framevitals/profiler.py",
  },
  {
    id: "rolesSummary",
    name: "Column Roles",
    category: "Profiling",
    oneLiner: "Infers semantic roles for each column.",
    what: "Tags ID-like, sensitive, target-candidate, high-cardinality, constant, severely-missing, and meaningful numeric columns.",
    how: "Combines column-name, dtype, cardinality, uniqueness, and missingness heuristics.",
    algorithms: ["rule engine", "cardinality heuristics"],
    source: "src/framevitals/column_roles.py",
  },
  {
    id: "datasetSignals",
    name: "Dataset Signals",
    category: "Profiling",
    oneLiner: "Summarizes dataset-level conditions that affect analysis.",
    what: "Produces flags for dates, text, high cardinality, severe missingness, potential leakage, imbalance, and related conditions.",
    how: "Aggregates profile and column-role evidence through deterministic predicates.",
    algorithms: ["rule predicates"],
    source: "src/framevitals/dataset_signals.py",
  },
  {
    id: "analysis_selector",
    name: "Analysis Selector",
    category: "Profiling",
    oneLiner: "Plans which diagnostics are relevant for a dataset.",
    what: "Marks analyses as selected, recommended, or skipped for the chosen mode and target.",
    how: "Evaluates the analysis inventory against dataset signals and analysis mode.",
    algorithms: ["signal-gated selection"],
    source: "src/framevitals/analysis_selector.py",
  },
  {
    id: "statistics",
    name: "Deep Statistics",
    category: "Statistics",
    oneLiner: "Univariate and bivariate statistical diagnostics.",
    what: "Runs normality, distribution-fit, bootstrap, association, and effect-size analyses.",
    how: "Uses SciPy and statsmodels tests with bounded pairwise analysis budgets.",
    algorithms: ["Shapiro-Wilk", "D'Agostino", "Anderson-Darling", "bootstrap", "Cramer's V"],
    source: "src/framevitals/deep_statistics_v2.py",
  },
  {
    id: "anomalies",
    name: "Anomaly Ensemble",
    category: "Quality",
    oneLiner: "Combines multiple detectors into row-level anomaly scores.",
    what: "Scores suspicious rows and reports the strongest anomaly candidates.",
    how: "Normalizes scores from available statistical and ML detectors before combining them.",
    algorithms: ["IsolationForest", "LOF", "EllipticEnvelope", "MAD", "Mahalanobis", "ECOD", "COPOD"],
    source: "src/framevitals/anomaly_ensemble.py",
  },
  {
    id: "ml-lab",
    name: "Model Leaderboard",
    category: "ML",
    oneLiner: "Builds a small cross-validated supervised-learning benchmark.",
    what: "Compares baseline, linear, nearest-neighbor, tree, boosting, and optional gradient-boosting models.",
    how: "Uses shared preprocessing plus cross-validation appropriate to classification or regression.",
    algorithms: ["scikit-learn", "XGBoost", "LightGBM"],
    source: "src/framevitals/model_leaderboard.py",
  },
  {
    id: "targetAnalysis",
    name: "Target Analysis",
    category: "ML",
    oneLiner: "Inspects task type and target health.",
    what: "Detects classification versus regression, imbalance, skew, outliers, and target warnings.",
    how: "Uses dtype, cardinality, distribution, and role heuristics.",
    algorithms: ["pandas", "scipy.stats"],
    source: "src/framevitals/target_analyzer.py",
  },
  {
    id: "featureImportance",
    name: "Feature Importance",
    category: "ML",
    oneLiner: "Ranks features by predictive contribution.",
    what: "Produces a global feature ranking for target-aware analysis.",
    how: "Uses model-native, SHAP, or permutation-based importance when available.",
    algorithms: ["SHAP", "permutation importance"],
    source: "src/framevitals/feature_importance.py",
  },
  {
    id: "baselineModel",
    name: "Baseline Model",
    category: "ML",
    oneLiner: "Provides an uninformed reference model.",
    what: "Makes it clear whether trained models beat a simple baseline.",
    how: "Uses DummyClassifier or DummyRegressor with cross-validation.",
    algorithms: ["scikit-learn"],
    source: "src/framevitals/baseline_model.py",
  },
  {
    id: "shap",
    name: "Explainability",
    category: "ML",
    oneLiner: "Explains the strongest model globally and per row.",
    what: "Produces global feature rankings and local contribution stories.",
    how: "Uses SHAP when supported and deterministic permutation importance as a fallback.",
    algorithms: ["SHAP", "permutation importance"],
    source: "src/framevitals/explainability.py",
  },
  {
    id: "timeseries",
    name: "Time Series",
    category: "Time",
    oneLiner: "Diagnoses temporal structure when date-like data is present.",
    what: "Checks frequency, stationarity, decomposition, autocorrelation, and a forecast preview.",
    how: "Uses date-confidence heuristics plus statsmodels time-series routines.",
    algorithms: ["ADF", "KPSS", "STL", "ACF", "PACF", "Holt-Winters"],
    source: "src/framevitals/time_series.py",
  },
  {
    id: "text",
    name: "Text Profile",
    category: "Text",
    oneLiner: "Profiles free-text columns without requiring a full NLP stack.",
    what: "Reports vocabulary, lengths, n-grams, patterns, lightweight language/sentiment signals, and a document map.",
    how: "Uses regex tokenization, TF-IDF, stopword filtering, and truncated SVD.",
    algorithms: ["TF-IDF", "TruncatedSVD", "regex"],
    source: "src/framevitals/text_profile.py",
  },
  {
    id: "drift",
    name: "Drift Analysis",
    category: "Drift",
    oneLiner: "Measures distribution changes between reference and current data.",
    what: "Reports per-column drift severity plus an overall verdict.",
    how: "Combines PSI with numeric KS tests or categorical chi-square tests.",
    algorithms: ["PSI", "Kolmogorov-Smirnov", "chi-square"],
    source: "src/framevitals/drift_analysis.py",
  },
  {
    id: "diagnostics",
    name: "Model Diagnostics",
    category: "ML",
    oneLiner: "Inspects the behavior of a fitted classification or regression model.",
    what: "Produces residual, calibration, and task-specific diagnostic evidence.",
    how: "Uses holdout predictions and standard classification/regression diagnostic metrics.",
    algorithms: ["scikit-learn", "numpy"],
    source: "src/framevitals/model_diagnostics.py",
  },
  {
    id: "multicollinearity",
    name: "Multicollinearity",
    category: "ML",
    oneLiner: "Finds redundant numeric predictors.",
    what: "Reports VIF and groups of strongly redundant variables.",
    how: "Measures how well each numeric feature is explained by its peers.",
    algorithms: ["VIF", "correlation grouping"],
    source: "src/framevitals/multicollinearity.py",
  },
  {
    id: "targetLeakage",
    name: "Target Leakage",
    category: "ML",
    oneLiner: "Flags features that may reveal the target improperly.",
    what: "Surfaces direct matches, suspicious correlations, and target-like naming patterns with severity evidence.",
    how: "Combines direct-value, correlation, and name-similarity checks.",
    algorithms: ["correlation", "name similarity", "direct-match checks"],
    source: "src/framevitals/target_leakage.py",
  },
  {
    id: "segments",
    name: "Segment Analysis",
    category: "ML",
    oneLiner: "Compares meaningful subgroups in the data.",
    what: "Builds low-cardinality slices and target-by-segment summaries.",
    how: "Uses bounded pandas group-by analysis on interpretable segment candidates.",
    algorithms: ["pandas group-by"],
    source: "src/framevitals/segment_analysis.py",
  },
  {
    id: "cleaning",
    name: "Cleaning",
    category: "Cleaning",
    oneLiner: "Creates a conservative cleaned dataset when artifacts are enabled.",
    what: "Handles duplicates, missing values, and selected quality issues while recording what changed.",
    how: "Uses deterministic pandas/scikit-learn transformations and keeps filesystem output opt-in for the public API.",
    algorithms: ["pandas", "scikit-learn"],
    source: "src/framevitals/cleaner.py",
  },
  {
    id: "ai-report",
    name: "AI Narrative",
    category: "AI",
    oneLiner: "Turns computed diagnostics into a grounded narrative.",
    what: "Summarizes quality, ML readiness, risks, and suggested next steps.",
    how: "Uses OpenRouter or Ollama when configured and falls back to deterministic evidence-based text.",
    algorithms: ["LLM", "deterministic fallback"],
    source: "src/framevitals/ai_insights.py",
  },
  {
    id: "ask",
    name: "Ask Anything",
    category: "AI",
    oneLiner: "Answers dataset questions using computed FrameVitals evidence.",
    what: "Provides agentic Q&A over the cached analysis result.",
    how: "Runs a planner/executor/critic/writer loop over scoped analysis tools with deterministic fallback.",
    algorithms: ["tool-calling agent", "RAG"],
    source: "src/framevitals/ai_agent.py",
  },
];

export function moduleById(id: string): ModuleRegistryEntry | undefined {
  return MODULES.find((module) => module.id === id);
}
