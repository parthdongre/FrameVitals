/**
 * Tightened payload types for the v3 rebuild.
 *
 * The pre-existing `DashboardTelemetry` in `mockTelemetry.ts` describes most
 * top-level keys with `any` to keep the legacy code compiling. We don't
 * touch that file — instead we add narrow interfaces here for the niche
 * sections the new tabs render. Tabs cast through these via `safeObj` /
 * `safeArr` from `@/lib/safe`.
 */

export type {
  DashboardTelemetry,
  AnalysisMode,
  SignalSeverity,
  AnalysisPlanItem,
  AnalysisPlanSummary,
  AnalysisSelectionTelemetry,
  DataTypeStat,
  MissionMetric,
  MissionSignal,
  DistributionPoint,
  DistributionTelemetry,
} from "./mockTelemetry";

/* --------------------------------------------------------------------------
 * Charts
 * -------------------------------------------------------------------------- */

export interface ChartItem {
  title: string;
  type: string;
  description?: string;
  column?: string;
  path: string;
  planner_reason?: string;
}

/* --------------------------------------------------------------------------
 * Signals (extended — backend optionally tags status / icon)
 * -------------------------------------------------------------------------- */

export interface SignalItem {
  name: string;
  severity: "low" | "medium" | "high" | "critical" | string;
  evidence: string;
  recommendation: string;
  status?: string;
  icon?: string;
}

/* --------------------------------------------------------------------------
 * Anomaly ensemble v2
 * -------------------------------------------------------------------------- */

export interface AnomalyTopRow {
  index?: number;
  score?: number;
  contributors?: Record<string, number>;
  [key: string]: unknown;
}

export interface AnomaliesV2 {
  available: boolean;
  n_rows_scored?: number;
  used_columns?: string[];
  detectors_run?: string[];
  threshold?: number;
  flagged_count?: number;
  ensemble_summary?: Record<string, unknown>;
  top_rows?: AnomalyTopRow[];
}

/* --------------------------------------------------------------------------
 * Model leaderboard
 * -------------------------------------------------------------------------- */

export interface LeaderboardRow {
  model: string;
  primary_metric_value?: number;
  metrics?: Record<string, number>;
  rank?: number;
  fold_scores?: number[];
  notes?: string;
}

export interface ModelLeaderboard {
  available: boolean;
  task_type?: "classification" | "regression" | string;
  target_column?: string;
  primary_metric?: string;
  n_rows?: number;
  n_features?: number;
  leaderboard?: LeaderboardRow[];
  winner?: LeaderboardRow;
  dropped_columns?: string[];
}

/* --------------------------------------------------------------------------
 * Explainability (SHAP / permutation)
 * -------------------------------------------------------------------------- */

export interface GlobalImportanceItem {
  feature: string;
  importance: number;
  direction?: "positive" | "negative" | "mixed";
}

export interface PerRowStory {
  index?: number;
  prediction?: number | string;
  base_value?: number;
  contributions?: Array<{ feature: string; value?: unknown; impact: number }>;
  narrative?: string;
}

export interface Explainability {
  available: boolean;
  model?: string;
  task_type?: string;
  method?: "shap" | "permutation" | string;
  global_importance?: GlobalImportanceItem[];
  per_row_stories?: PerRowStory[];
  summary_chart_path?: string;
  n_test_rows_explained?: number;
  permutation_importance?: GlobalImportanceItem[];
  errors?: string[];
}

/* --------------------------------------------------------------------------
 * Time-series
 * -------------------------------------------------------------------------- */

export interface TimeSeriesDecomposition {
  trend?: number[];
  seasonal?: number[];
  resid?: number[];
  observed?: number[];
  index?: string[];
}

export interface AcfPacf {
  acf?: number[];
  pacf?: number[];
  lags?: number[];
}

export interface StationarityResult {
  test?: string;
  statistic?: number;
  pvalue?: number;
  is_stationary?: boolean;
}

export interface ForecastResult {
  method?: string;
  horizon?: number;
  index?: string[];
  forecast?: number[];
  lower?: number[];
  upper?: number[];
}

export interface TimeSeries {
  available: boolean;
  detected_date_column?: string;
  frequency?: string;
  period_estimate?: number;
  stationarity?: StationarityResult;
  decomposition?: TimeSeriesDecomposition;
  autocorrelation?: AcfPacf;
  forecast?: ForecastResult;
}

/* --------------------------------------------------------------------------
 * Text profile
 * -------------------------------------------------------------------------- */

export interface LsaPoint {
  x: number;
  y: number;
  index?: number;
  preview?: string;
}

export interface TextProfileColumn {
  ngrams?: { text: string; count: number }[];
  vocabulary?: number;
  avg_length?: number;
  patterns?: Record<string, number>;
  lsa?: { points?: LsaPoint[]; topics?: { weight: number; tokens: string[] }[] };
}

export interface TextProfile {
  available: boolean;
  detected_columns?: string[];
  profiled_columns?: string[];
  stopwords_source?: string;
  profiles?: Record<string, TextProfileColumn>;
}

/* --------------------------------------------------------------------------
 * Cleaning
 * -------------------------------------------------------------------------- */

export interface CleaningAction {
  step?: string;
  detail?: string;
  affected?: number;
  [key: string]: unknown;
}

export interface CleaningPayload {
  actions?: CleaningAction[];
  missing_before?: Record<string, number> | number;
  missing_after?: Record<string, number> | number;
  duplicates_before?: number;
  duplicates_after?: number;
  before_health?: { score?: number; label?: string };
  after_health?: { score?: number; label?: string };
  output_path?: string;
}

/* --------------------------------------------------------------------------
 * Ask Anything (agentic Q&A)
 * -------------------------------------------------------------------------- */

export interface AskTraceStep {
  step?: string;
  tool?: string;
  args?: Record<string, unknown>;
  result?: unknown;
}

export interface AskResponse {
  question: string;
  dataset_id: string;
  source: string;
  answer: string;
  trace?: AskTraceStep[] | Record<string, unknown>;
}

/* --------------------------------------------------------------------------
 * Health probe (`/api/health`)
 * -------------------------------------------------------------------------- */

export interface HealthResponse {
  flask: boolean;
  ollama_reachable?: boolean;
  openrouter_configured?: boolean;
  cached_analyses?: number;
  pdf_jobs?: number;
  version?: string;
}

/* --------------------------------------------------------------------------
 * Drift (POST /api/compare and /api/compare-self)
 * -------------------------------------------------------------------------- */

export interface DriftColumnResult {
  column: string;
  type?: string;
  test?: string;
  statistic?: number;
  pvalue?: number;
  psi?: number;
  severity?: "stable" | "minor" | "moderate" | "severe" | string;
  reference_summary?: Record<string, unknown>;
  current_summary?: Record<string, unknown>;
  histogram?: { bins?: string[]; reference?: number[]; current?: number[] };
}

export interface DriftReport {
  reference_filename?: string;
  current_filename?: string;
  split_by?: string;
  split_ratio?: number;
  columns?: DriftColumnResult[];
  summary?: {
    n_columns?: number;
    n_drifted?: number;
    severity_counts?: Record<string, number>;
  };
}

/* --------------------------------------------------------------------------
 * Report PDF status
 * -------------------------------------------------------------------------- */

export interface ReportStatusResponse {
  status: "queued" | "running" | "ready" | "failed" | "missing" | "pending" | string;
  ready: boolean;
  pdf_path?: string;
  error?: string | null;
  downloadUrl?: string;
  dataset_id?: string;
}

/* --------------------------------------------------------------------------
 * Default-shape factories. Tabs use these so destructuring on an empty
 * payload section yields a valid object instead of throwing.
 * -------------------------------------------------------------------------- */

export const defaultAnomaliesV2 = (): AnomaliesV2 => ({ available: false });
export const defaultLeaderboard = (): ModelLeaderboard => ({ available: false });
export const defaultExplainability = (): Explainability => ({ available: false });
export const defaultTimeSeries = (): TimeSeries => ({ available: false });
export const defaultTextProfile = (): TextProfile => ({ available: false });
export const defaultCleaning = (): CleaningPayload => ({});
