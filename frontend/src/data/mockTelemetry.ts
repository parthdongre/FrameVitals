export type AnalysisMode = "quick" | "standard" | "deep" | "research";

export type SignalSeverity = "low" | "medium" | "high" | "critical";

export interface AnalysisPlanItem {
  id?: string;
  name: string;
  category?: string;
  priority?: string;
  modes?: string[];
  outputs?: string[];
}

export interface AnalysisPlanSummary {
  selected_count: number;
  recommended_count: number;
  skipped_count: number;
}

export interface AnalysisSelectionTelemetry {
  selectedAnalyses: AnalysisPlanItem[];
  recommendedAnalyses: AnalysisPlanItem[];
  skippedAnalyses: AnalysisPlanItem[];
  summary: AnalysisPlanSummary;
}

export interface DataTypeStat {
  label: string;
  count: number;
}

export interface MissionMetric {
  label: string;
  value: string;
  hint: string;
}

export interface MissionSignal {
  name: string;
  severity: SignalSeverity;
  evidence: string;
  recommendation: string;
}

export interface DistributionPoint {
  label: string;
  value: number;
}

export interface DistributionTelemetry {
  title: string;
  subtitle: string;
  points: DistributionPoint[];
  mean: number;
  stdDev: number;
  min: number;
  max: number;
  totalRows: number;
}

export interface DashboardTelemetry {
  id: string;
  filename: string;
  analysisMode: AnalysisMode;
  selectedTargetColumn?: string | null;
  targetCandidates?: string[];
  rows: number;
  columns: number;
  fileSize: string;
  lastScan: string;
  dataTypes: DataTypeStat[];
  metrics: MissionMetric[];
  signals: MissionSignal[];
  distribution: DistributionTelemetry;
  analysisSelection?: AnalysisSelectionTelemetry;
  profile?: any;
  rolesSummary?: any;
  columnRoles?: Record<string, any>;
  deepStatistics?: any;
  targetAnalysis?: any;
  featureImportance?: any;
  baselineModel?: any;
  targetLeakage?: any;
  multicollinearity?: any;
  modelDiagnostics?: any;
  segmentAnalysis?: any;
  cleaning?: any;
  charts?: any[];
  aiReport?: any;
  health?: any;
  mlReadiness?: any;
  deepStatisticsV2?: any;
  anomaliesV2?: any;
  modelLeaderboard?: any;
  explainability?: any;
  timeSeries?: any;
  textProfile?: any;
  analysisDurationMs?: number;
  downloadLinks?: {
    cleaned: string;
    report: string;
    reportReady?: boolean;
    reportStatus?: string;
  };
}

export const mockTelemetry: DashboardTelemetry = {
  id: "DS-0001",
  filename: "sample_dataset.csv",
  analysisMode: "standard",
  selectedTargetColumn: null,
  targetCandidates: ["target", "label", "outcome"],
  rows: 128_420,
  columns: 24,
  fileSize: "84.2 MB",
  lastScan: "T+04:12:09",
  dataTypes: [
    { label: "Numeric", count: 14 },
    { label: "Categorical", count: 6 },
    { label: "Temporal", count: 2 },
    { label: "Boolean", count: 2 },
  ],
  metrics: [
    { label: "Data Completeness", value: "99.2%", hint: "No schema drift detected" },
    { label: "Processing Time", value: "12 ms", hint: "Pipeline completed within the expected range" },
    { label: "Feature Stability", value: "0.18", hint: "Low variance across the sample" },
  ],
  signals: [
    {
      name: "Missing value concentration",
      severity: "medium",
      evidence: "14% missingness is concentrated in a small subset of columns.",
      recommendation: "Review imputation strategy before modelling.",
    },
    {
      name: "Potential leakage risk",
      severity: "high",
      evidence: "A duplicate target surrogate closely matches the label distribution.",
      recommendation: "Exclude target-like columns before model training.",
    },
    {
      name: "Outlier pattern",
      severity: "low",
      evidence: "3.2% of records show transient variance spikes.",
      recommendation: "Monitor with a rolling z-score threshold.",
    },
  ],
  distribution: {
    title: "Feature Value Distribution",
    subtitle: "Histogram of observed numeric values across the sample.",
    points: [
      { label: "0-5", value: 3 },
      { label: "5-10", value: 7 },
      { label: "10-15", value: 14 },
      { label: "15-20", value: 22 },
      { label: "20-25", value: 31 },
      { label: "25-30", value: 45 },
      { label: "30-35", value: 58 },
      { label: "35-40", value: 71 },
      { label: "40-45", value: 63 },
      { label: "45-50", value: 49 },
      { label: "50-55", value: 32 },
      { label: "55-60", value: 19 },
    ],
    mean: 42.7,
    stdDev: 8.4,
    min: 12.3,
    max: 67.1,
    totalRows: 128_420,
  },
  analysisSelection: {
    selectedAnalyses: [
      { name: "Dataset Profiling", category: "Profiling", priority: "essential" },
      { name: "Missing Value Analysis", category: "Data Quality", priority: "essential" },
      { name: "Correlation Analysis", category: "Relationships", priority: "high" },
      { name: "Cleaning Analysis", category: "Cleaning", priority: "high" },
    ],
    recommendedAnalyses: [
      { name: "Target Column Analysis", category: "Machine Learning", priority: "high" },
      { name: "Feature Importance Analysis", category: "Machine Learning", priority: "high" },
    ],
    skippedAnalyses: [],
    summary: {
      selected_count: 4,
      recommended_count: 2,
      skipped_count: 0,
    },
  },
};