import { useState } from "react";
import { Eyebrow, PageTitle, Section, SectionHeader } from "@/components/site/SiteShell";
import { cn } from "@/lib/utils";

interface ModuleEntry {
  id: string;
  name: string;
  category: "Profiling" | "Statistics" | "Quality" | "ML" | "Time" | "Text" | "Drift" | "AI";
  oneLiner: string;
  what: string;
  how: string;
  algorithms: string[];
  inputs: string;
  output: string;
  source: string;
}

const MODULES: ModuleEntry[] = [
  // ---- Profiling -----------------------------------------------------------
  {
    id: "profiler",
    name: "Profiler",
    category: "Profiling",
    oneLiner: "Builds the structural snapshot of a dataset.",
    what:
      "Computes shape, dtypes, missing counts, duplicate rows, numeric/categorical/date column lists, descriptive statistics for numeric columns, top values for categorical columns, a Pearson correlation matrix, and a 15-row preview.",
    how:
      "Uses pandas dtypes for type inference, `df.describe()` for numeric stats, `value_counts(dropna=False)` for categorical tops, and a heuristic date sniff (parse the first 25 values with `pd.to_datetime`, accept the column when ≥70% parse successfully).",
    algorithms: ["pandas dtype inference", "describe()", "to_datetime sniffing", "Pearson correlation"],
    inputs: "DataFrame.",
    output: "Profile dict consumed by every other module downstream.",
    source: "modules/profiler.py",
  },
  {
    id: "column_roles",
    name: "Column Roles",
    category: "Profiling",
    oneLiner: "Tags each column with semantic roles.",
    what:
      "Assigns tags like id_like, sensitive, target_candidate, high_cardinality, constant, severe_missing, numeric_meaningful so downstream modules can make smarter decisions.",
    how:
      "Combines column-name keyword matching, cardinality ratios, missing percentages, and dtype checks into a small rule engine.",
    algorithms: ["keyword matching", "cardinality + missingness rules"],
    inputs: "DataFrame.",
    output: "Per-column role list + a roles_summary dict.",
    source: "modules/column_roles.py",
  },
  {
    id: "dataset_signals",
    name: "Dataset Signals",
    category: "Profiling",
    oneLiner: "Boolean flags about the dataset as a whole.",
    what:
      "Computes flags like has_potential_leakage, has_class_imbalance, has_high_cardinality, has_dates, has_text, has_severe_missing.",
    how:
      "Aggregates the profiler + column-roles output through small rule predicates.",
    algorithms: ["rule predicates"],
    inputs: "DataFrame + profile.",
    output: "Boolean signal dict.",
    source: "modules/dataset_signals.py",
  },
  {
    id: "analysis_selector",
    name: "Analysis Selector",
    category: "Profiling",
    oneLiner: "Decides which analyses to actually run.",
    what:
      "Given the dataset signals + analysis mode, decides which downstream analyses to mark as selected, recommended, or skipped — keeps the pipeline focused.",
    how:
      "Looks up an analysis inventory and applies signal-driven gates per analysis.",
    algorithms: ["signal-gated selection"],
    inputs: "signals, mode, target_column.",
    output: "Plan with selected / recommended / skipped lists.",
    source: "modules/analysis_selector.py",
  },

  // ---- Statistics ----------------------------------------------------------
  {
    id: "deep_statistics_v2",
    name: "Deep Statistics v2",
    category: "Statistics",
    oneLiner: "Research-grade statistical battery per column and per pair.",
    what:
      "For every numeric column: descriptive stats, three independent outlier views, a normality verdict, a best-fit distribution, bootstrap CIs. For every categorical column: cardinality + entropy + top values. Bivariate panels for numeric/numeric, categorical/categorical, and binary/numeric.",
    how:
      "Shapiro-Wilk + D'Agostino + Anderson-Darling for normality with verdict consensus. AIC-based selection across {norm, lognorm, expon, gamma, weibull_min, beta} for distribution fit. BCa bootstrap (999 resamples, percentile fallback). Pearson/Spearman/Kendall with p-values, Cramér's V, point-biserial, Mann-Whitney U / Kruskal-Wallis. Pair budget capped at 20 per category.",
    algorithms: [
      "Shapiro-Wilk", "D'Agostino-Pearson", "Anderson-Darling", "AIC for distribution selection",
      "BCa bootstrap", "Pearson · Spearman · Kendall", "Cramér's V", "Point-biserial",
      "Mann-Whitney U", "Kruskal-Wallis H",
    ],
    inputs: "DataFrame.",
    output: "JSON-safe dict with numeric / categorical / bivariate sections.",
    source: "modules/deep_statistics_v2.py",
  },

  // ---- Quality -------------------------------------------------------------
  {
    id: "health_score",
    name: "Health Score",
    category: "Quality",
    oneLiner: "Composite 0-100 quality score.",
    what:
      "Produces an overall health score plus four sub-scores: completeness, consistency, uniqueness, outlier safety. Comes with a label (Excellent / Good / Moderate / Poor / Critical).",
    how:
      "Weighted combination of missing percentage, duplicate percentage, constant column count, and an IQR-based outlier rate, normalized into the 0-100 range.",
    algorithms: ["weighted composite", "IQR outlier counting"],
    inputs: "DataFrame + profile.",
    output: "{ overall_score, label, components, details }",
    source: "modules/health_score.py",
  },
  {
    id: "ml_readiness",
    name: "ML Readiness",
    category: "Quality",
    oneLiner: "Heuristic 0-100 modelling readiness.",
    what:
      "Estimates how ready a dataset is for supervised learning. Penalizes missingness, ID-likes, very high cardinality, severely skewed distributions.",
    how:
      "Per-column heuristic + weighted aggregation. Returns recommendations (e.g. encode high-cardinality, impute missing, drop constants).",
    algorithms: ["heuristic scoring", "weighted aggregation"],
    inputs: "DataFrame.",
    output: "{ score, label, recommendations }",
    source: "modules/ml_readiness.py",
  },
  {
    id: "advanced_indicators",
    name: "Advanced Indicators",
    category: "Quality",
    oneLiner: "Column utility, fairness, freshness, leakage hints.",
    what:
      "Scores each column's analytical usefulness, IQR-based row anomalies, fairness keyword scan (gender / age / race-like names), date-coverage freshness, and lightweight column-pair leakage hints (95% match on non-missing rows).",
    how:
      "Per-column rule scoring. IQR flags across all numeric columns. Sensitive-keyword regex. String-equality match-ratio between non-missing column pairs.",
    algorithms: ["IQR flags", "regex keyword matching", "match-ratio leakage"],
    inputs: "DataFrame.",
    output: "{ column_utility, anomalies, fairness, freshness, leakage }",
    source: "modules/advanced_indicators.py",
  },
  {
    id: "anomaly_ensemble",
    name: "Anomaly Ensemble",
    category: "Quality",
    oneLiner: "Seven-detector ensemble for multivariate anomalies.",
    what:
      "Each detector emits a [0, 1] score per row. The ensemble score is the per-row mean across all available detectors. Rows with ensemble ≥ 0.6 are flagged.",
    how:
      "Median-impute and standard-scale the numeric matrix, then run: IsolationForest (sklearn), LocalOutlierFactor (sklearn), EllipticEnvelope (sklearn), Robust z-score via MAD (numpy), Mahalanobis distance with MinCovDet (sklearn), ECOD (pyod), COPOD (pyod). Each raw score is min-max normalized before averaging.",
    algorithms: ["IsolationForest", "LOF", "EllipticEnvelope", "MAD-z", "Mahalanobis", "ECOD", "COPOD"],
    inputs: "DataFrame.",
    output: "Per-row score table + summary stats + top-K rows.",
    source: "modules/anomaly_ensemble.py",
  },

  // ---- ML ------------------------------------------------------------------
  {
    id: "model_leaderboard",
    name: "Model Leaderboard",
    category: "ML",
    oneLiner: "CV leaderboard of 7-8 models with calibration.",
    what:
      "Trains a small leaderboard against the user's target column. Classification: Dummy, LogisticRegression, KNN, RandomForest, GradientBoosting, XGBoost, LightGBM. Regression: Dummy, Ridge, Lasso, KNN, RandomForest, GradientBoosting, XGBoost, LightGBM.",
    how:
      "5-fold StratifiedKFold for classification (auto-shrinks to min class size); KFold for regression. Shared sklearn ColumnTransformer (median impute + scale for numeric, mode impute + one-hot for categorical). Refits the winner on a 75/25 holdout and reports Brier score for binary calibration or residual stats for regression.",
    algorithms: ["StratifiedKFold", "KFold", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "Brier score"],
    inputs: "DataFrame, target_column, optional task_type.",
    output: "Leaderboard rows + winner card + holdout metrics.",
    source: "modules/model_leaderboard.py",
  },
  {
    id: "explainability",
    name: "Explainability (SHAP)",
    category: "ML",
    oneLiner: "Global + per-row attribution for the leaderboard winner.",
    what:
      "For the winning model: a global feature ranking, three per-row 'stories' showing the most influential features for the most-explained rows, and a saved beeswarm summary plot.",
    how:
      "shap.TreeExplainer for tree models (RF, GB, XGB, LGBM). shap.LinearExplainer for linear models (LogReg, Ridge, Lasso). sklearn permutation_importance always runs as a cross-check / fallback. One-hot expansions are collapsed back to original feature names.",
    algorithms: ["SHAP TreeExplainer", "SHAP LinearExplainer", "Permutation importance"],
    inputs: "DataFrame + leaderboard winner.",
    output: "Global ranking + per-row stories + beeswarm PNG.",
    source: "modules/explainability.py",
  },
  {
    id: "ml_preprocessing",
    name: "ML Preprocessing",
    category: "ML",
    oneLiner: "Single source of truth for ML feature prep.",
    what:
      "Drops rows where target is missing, removes ID-likes, constants, all-empty columns, high-cardinality non-numeric columns, and time-like text. Builds the shared sklearn ColumnTransformer used across modules.",
    how:
      "Keyword + cardinality + missingness heuristics. Pipelines: median + StandardScaler for numeric, most-frequent + OneHotEncoder(handle_unknown='ignore') for categorical.",
    algorithms: ["pre-flight column filtering", "ColumnTransformer"],
    inputs: "DataFrame, target_column.",
    output: "X, y, numeric_features, categorical_features, dropped_columns, warnings.",
    source: "modules/ml_preprocessing.py",
  },

  // ---- Time ----------------------------------------------------------------
  {
    id: "time_series",
    name: "Time-series",
    category: "Time",
    oneLiner: "Auto-detected time-series mini-pipeline.",
    what:
      "When a confidently date-like column is detected, runs frequency inference, stationarity tests, STL decomposition, autocorrelation, and a forecast preview.",
    how:
      "Confidence-scored date detection that rejects low-cardinality numeric columns. Frequency from the median inter-arrival in seconds. ADF + KPSS with a consensus verdict. STL with FFT-detected period. ACF / PACF up to min(40, n//5). Holt-Winters (additive) with seasonal_periods set to the detected period; naive last-value baseline for honest comparison.",
    algorithms: ["ADF", "KPSS", "STL", "ACF / PACF", "Holt-Winters"],
    inputs: "DataFrame, optional target_column.",
    output: "Decomposition previews, ACF / PACF, forecast vs naive metrics.",
    source: "modules/time_series.py",
  },

  // ---- Text ----------------------------------------------------------------
  {
    id: "text_profile",
    name: "Text / NLP profile",
    category: "Text",
    oneLiner: "Per-column linguistic profile for free text.",
    what:
      "When a column looks like free text (multi-token rows + average length ≥ 8), produces token / vocab / lexical-diversity stats, top n-grams, regex pattern hits, language guess, sentiment-lite, and a 2-D LSA scatter.",
    how:
      "Regex tokenization, NLTK English stopwords (with a built-in fallback list), top 15 unigrams + 12 bigrams. Six probes for emails / URLs / phones / mentions / hashtags / monetary. Language inferred from ASCII fraction + stopword overlap. Sentiment from a small positive/negative seed lexicon. TF-IDF (max 1024 features, 1-2 grams) + TruncatedSVD(2) for the document map.",
    algorithms: ["TF-IDF", "TruncatedSVD", "NLTK stopwords", "regex pattern probes"],
    inputs: "DataFrame.",
    output: "{ detected_columns, profiles[col]: { ... } }",
    source: "modules/text_profile.py",
  },

  // ---- Drift ---------------------------------------------------------------
  {
    id: "drift_analysis",
    name: "Drift / Compare",
    category: "Drift",
    oneLiner: "Compare two datasets, or one dataset across time.",
    what:
      "Quantifies how much each column has shifted between a reference and a current dataset. Numeric columns get PSI + KS + mean shift; categorical columns get PSI + chi-square + new/missing categories.",
    how:
      "PSI on 10 quantile bins of the reference (numeric) or category proportions (categorical), with Laplace smoothing. Severity buckets: stable < 0.10, minor < 0.25, moderate < 0.50, severe ≥ 0.50. ks_2samp for numeric two-sample. chi2_contingency for categorical. split_by_date(df, col, ratio) to compare a single dataset chronologically.",
    algorithms: ["PSI", "Kolmogorov-Smirnov", "Chi-square", "Laplace smoothing"],
    inputs: "Two DataFrames OR one DataFrame + date column.",
    output: "Severity counts, overall verdict, per-column drift rows.",
    source: "modules/drift_analysis.py",
  },

  // ---- AI ------------------------------------------------------------------
  {
    id: "ai_agent",
    name: "AI Agent",
    category: "AI",
    oneLiner: "Planner → Executor → Critic → Writer over local Ollama.",
    what:
      "Answers free-form questions about the analyzed dataset. The planner picks 1-3 tool calls; the executor runs them; the critic checks for hallucinations and can request a single repair round; the writer composes a final markdown answer.",
    how:
      "All LLM calls use Ollama JSON-mode with Pydantic validation on the responses. Three-tier fallback: Ollama → OpenRouter → a deterministic heuristic writer that quotes RAG-retrieved facts directly.",
    algorithms: ["Pydantic JSON validation", "Tool-calling agent loop", "Three-tier fallback"],
    inputs: "Question, DataFrame, analysis_result.",
    output: "{ source, answer (markdown), trace }",
    source: "modules/ai_agent.py",
  },
  {
    id: "rag_index",
    name: "RAG Fact Index",
    category: "AI",
    oneLiner: "Atomic-fact retrieval over the analysis result.",
    what:
      "Flattens an analysis result dict into ~600 atomic facts, then retrieves the top-k most relevant for any question.",
    how:
      "Walks the result dict and yields one fact per leaf path. Preferred backend: Ollama embeddings (nomic-embed-text). Fallback: TF-IDF cosine similarity. Identical API regardless of backend.",
    algorithms: ["Path flattening", "Ollama embeddings", "TF-IDF cosine"],
    inputs: "analysis_result, question, k.",
    output: "Top-k Fact records.",
    source: "modules/rag_index.py",
  },
  {
    id: "agent_tools",
    name: "Agent Tools",
    category: "AI",
    oneLiner: "Eight typed tools the LLM can call.",
    what:
      "get_section, list_columns, column_summary, run_query, get_top_anomalies, get_leaderboard, get_explainability_top, search_facts.",
    how:
      "Each tool is a typed Python function returning a JSON-safe dict. Dispatch goes through a single run_tool(name, args, ctx) entry. Outputs are size-capped for prompt-budget safety.",
    algorithms: ["Typed registry", "Size capping"],
    inputs: "Tool name + args + AgentContext.",
    output: "JSON-safe dict.",
    source: "modules/agent_tools.py",
  },
  {
    id: "safe_pandas",
    name: "Safe Pandas Sandbox",
    category: "AI",
    oneLiner: "AST-allowlist sandbox for LLM-submitted pandas expressions.",
    what:
      "Lets the LLM compute things like df.groupby('x')['y'].mean().head(5) without giving it generic Python execution. Blocks __import__, open, lambda, exec, eval, dunders, and any attribute not on a manually curated list.",
    how:
      "Parses the expression with ast.parse, walks the tree to confirm every node, name, and attribute is on the allow-list, then evaluates inside a builtins-stripped namespace with only df, np, pd, and a small set of safe functions.",
    algorithms: ["AST allowlist", "Builtins-stripped eval"],
    inputs: "Expression string + DataFrame.",
    output: "{ ok, result, error, expression }",
    source: "modules/safe_pandas.py",
  },
];

const CATEGORIES: { id: ModuleEntry["category"] | "All"; label: string }[] = [
  { id: "All", label: "All" },
  { id: "Profiling", label: "Profiling" },
  { id: "Quality", label: "Quality" },
  { id: "Statistics", label: "Statistics" },
  { id: "ML", label: "ML" },
  { id: "Time", label: "Time-series" },
  { id: "Text", label: "Text" },
  { id: "Drift", label: "Drift" },
  { id: "AI", label: "AI" },
];

export function ModulesPage() {
  const [filter, setFilter] = useState<ModuleEntry["category"] | "All">("All");
  const filtered = filter === "All" ? MODULES : MODULES.filter((m) => m.category === filter);

  return (
    <>
      <Section className="pb-8 pt-6">
        <Eyebrow>Module reference</Eyebrow>
        <PageTitle
          subtitle={`Every analytical module in DataLens AI v3 — ${MODULES.length} in total — explained in plain language. Each entry covers what the module does, how it does it, the underlying algorithms, and where the source lives.`}
        >
          What's under the hood.
        </PageTitle>
      </Section>

      <Section className="py-0">
        <div className="-mx-6 mb-12 flex gap-2 overflow-x-auto px-6 sm:mx-0 sm:px-0">
          {CATEGORIES.map((c) => {
            const active = filter === c.id;
            const count = c.id === "All" ? MODULES.length : MODULES.filter((m) => m.category === c.id).length;
            return (
              <button
                key={c.id}
                onClick={() => setFilter(c.id)}
                className={cn(
                  "shrink-0 rounded-full border px-4 py-1.5 text-xs font-medium transition-colors duration-200",
                  active
                    ? "border-[var(--accent-line)] bg-[var(--accent-soft)] text-[var(--ink-1)]"
                    : "border-[var(--line)] text-[var(--ink-3)] hover:border-[var(--line-strong)] hover:text-[var(--ink-1)]",
                )}
              >
                {c.label}
                <span className="ml-2 text-[10px] text-[var(--ink-4)]">{count}</span>
              </button>
            );
          })}
        </div>

        <div className="grid gap-4 sm:gap-6 md:grid-cols-2">
          {filtered.map((m) => (
            <ModuleCard key={m.id} module={m} />
          ))}
        </div>
      </Section>
    </>
  );
}

function ModuleCard({ module }: { module: ModuleEntry }) {
  const [open, setOpen] = useState(false);
  return (
    <article className="card group flex flex-col p-6 transition-transform duration-300 hover:-translate-y-0.5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow">{module.category}</p>
          <h3 className="mt-2 text-xl font-semibold text-[var(--ink-1)]">{module.name}</h3>
          <p className="mt-2 text-sm leading-7 text-[var(--ink-2)]">{module.oneLiner}</p>
        </div>
      </div>

      <button
        onClick={() => setOpen((v) => !v)}
        className="mt-5 inline-flex w-fit items-center gap-2 text-xs font-medium text-[var(--accent)] hover:text-[var(--ink-1)]"
      >
        {open ? "Hide details" : "Read details"}
        <span className={cn("transition", open ? "rotate-90" : "")}>→</span>
      </button>

      {open ? (
        <div className="mt-5 space-y-5 border-t border-[var(--line)] pt-5 text-sm leading-7 text-[var(--ink-2)]">
          <DetailBlock label="What it does" body={module.what} />
          <DetailBlock label="How it works" body={module.how} />

          <div>
            <Eyebrow className="mb-2">Algorithms</Eyebrow>
            <div className="flex flex-wrap gap-1.5">
              {module.algorithms.map((a) => (
                <span
                  key={a}
                  className="rounded-full border border-[var(--line)] bg-[var(--bg-2)] px-2 py-0.5 text-[11px] text-[var(--ink-2)]"
                >
                  {a}
                </span>
              ))}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <DetailMeta label="Inputs" body={module.inputs} />
            <DetailMeta label="Output" body={module.output} />
          </div>

          <p className="font-mono text-[11px] text-[var(--ink-3)]">{module.source}</p>
        </div>
      ) : null}
    </article>
  );
}

function DetailBlock({ label, body }: { label: string; body: string }) {
  return (
    <div>
      <Eyebrow className="mb-2">{label}</Eyebrow>
      <p className="text-pretty text-sm leading-7 text-[var(--ink-2)]">{body}</p>
    </div>
  );
}

function DetailMeta({ label, body }: { label: string; body: string }) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--bg-2)] px-3 py-2.5">
      <p className="label-mono">{label}</p>
      <p className="mt-1 text-xs leading-6 text-[var(--ink-2)]">{body}</p>
    </div>
  );
}
