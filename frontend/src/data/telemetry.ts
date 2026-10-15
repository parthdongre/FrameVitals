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
  targetIntelligence?: any;
  qualityDiagnostics?: any;
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

