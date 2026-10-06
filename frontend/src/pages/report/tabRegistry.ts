import { lazy, type ComponentType, type LazyExoticComponent } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  Brain,
  Eye,
  FileText,
  Gauge,
  Image as ImageIcon,
  LayoutGrid,
  LineChart,
  MessageSquare,
  Sigma,
  Sparkles,
  Stethoscope,
  Table as TableIcon,
  Timer,
  Waves,
} from "lucide-react";
import type { DashboardTelemetry } from "@/data/payload";
import { isPresent, safeArr } from "@/lib/safe";

export interface TabComponentProps {
  analysis: DashboardTelemetry;
}

export interface TabDef {
  id: string;
  label: string;
  Icon: ComponentType<{ className?: string; size?: number }>;
  hasData: (a: DashboardTelemetry) => boolean;
  Component: LazyExoticComponent<ComponentType<TabComponentProps>>;
}

export const REPORT_TABS: TabDef[] = [
  {
    id: "overview",
    label: "Overview",
    Icon: Gauge,
    hasData: () => true,
    Component: lazy(() => import("./OverviewTab")),
  },
  {
    id: "profile",
    label: "Profile",
    Icon: TableIcon,
    hasData: (a) => isPresent(a.profile),
    Component: lazy(() => import("./ProfileTab")),
  },
  {
    id: "statistics",
    label: "Statistics",
    Icon: Sigma,
    hasData: (a) => isPresent(a.deepStatisticsV2) || isPresent(a.deepStatistics),
    Component: lazy(() => import("./StatisticsTab")),
  },
  {
    id: "anomalies",
    label: "Anomalies",
    Icon: AlertTriangle,
    hasData: (a) => {
      const adv = (a as unknown as { advanced?: { anomalies?: unknown } }).advanced;
      return Boolean(a.anomaliesV2?.available) || isPresent(adv?.anomalies);
    },
    Component: lazy(() => import("./AnomaliesTab")),
  },
  {
    id: "ml-lab",
    label: "ML",
    Icon: Brain,
    hasData: (a) =>
      Boolean(a.modelLeaderboard?.available) || isPresent(a.targetAnalysis),
    Component: lazy(() => import("./MlLabTab")),
  },
  {
    id: "shap",
    label: "Explainability",
    Icon: LineChart,
    hasData: (a) => Boolean(a.explainability?.available),
    Component: lazy(() => import("./ShapTab")),
  },
  {
    id: "timeseries",
    label: "Time-series",
    Icon: Activity,
    hasData: (a) => Boolean(a.timeSeries?.available),
    Component: lazy(() => import("./TimeSeriesTab")),
  },
  {
    id: "text",
    label: "Text",
    Icon: FileText,
    hasData: (a) => Boolean(a.textProfile?.available),
    Component: lazy(() => import("./TextTab")),
  },
  {
    id: "drift",
    label: "Change",
    Icon: Waves,
    hasData: () => true,
    Component: lazy(() => import("./DriftTab")),
  },
  {
    id: "diagnostics",
    label: "Diagnostics",
    Icon: Stethoscope,
    hasData: (a) =>
      isPresent(a.modelDiagnostics) ||
      isPresent(a.multicollinearity) ||
      isPresent(a.targetLeakage),
    Component: lazy(() => import("./DiagnosticsTab")),
  },
  {
    id: "segments",
    label: "Segments",
    Icon: LayoutGrid,
    hasData: (a) => isPresent(a.segmentAnalysis),
    Component: lazy(() => import("./SegmentsTab")),
  },
  {
    id: "cleaning",
    label: "Transform",
    Icon: Sparkles,
    hasData: (a) => isPresent(a.cleaning),
    Component: lazy(() => import("./CleaningTab")),
  },
  {
    id: "charts",
    label: "Charts",
    Icon: ImageIcon,
    hasData: (a) =>
      safeArr(a.charts).length > 0 || Boolean(a.explainability?.summary_chart_path),
    Component: lazy(() => import("./ChartsTab")),
  },
  {
    id: "ai-report",
    label: "Narrative",
    Icon: BookOpen,
    hasData: () => true,
    Component: lazy(() => import("./AiReportTab")),
  },
  {
    id: "data-preview",
    label: "Data",
    Icon: Eye,
    hasData: (a) => safeArr((a.profile as any)?.preview).length > 0,
    Component: lazy(() => import("./DataPreviewTab")),
  },
  {
    id: "timings",
    label: "Timing",
    Icon: Timer,
    hasData: (a) => isPresent((a as any).timings_ms),
    Component: lazy(() => import("./TimingsTab")),
  },
  {
    id: "ask",
    label: "Ask",
    Icon: MessageSquare,
    hasData: () => true,
    Component: lazy(() => import("./AskAnythingTab")),
  },
];

export const TAB_IDS: string[] = REPORT_TABS.map((t) => t.id);

export function findTab(id: string | undefined): TabDef {
  if (!id) return REPORT_TABS[0];
  return REPORT_TABS.find((t) => t.id === id) ?? REPORT_TABS[0];
}
