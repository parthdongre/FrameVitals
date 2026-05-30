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
import { moduleById } from "@/data/moduleRegistry";

export interface TabComponentProps {
  analysis: DashboardTelemetry;
}

export interface HowThisWorks {
  title?: string;
  body: string;
  algorithms?: string[];
  source?: string;
}

export interface TabDef {
  /**
   * Stable url-friendly id used in `#/report?tab=<id>`.
   */
  id: string;
  /**
   * Visible tab label.
   */
  label: string;
  /**
   * Lucide icon component.
   */
  Icon: ComponentType<{ className?: string; size?: number }>;
  /**
   * Predicate that decides whether the tab has data. Returning false does
   * not hide the tab — the UI dims it and the tab body falls back to an
   * EmptyState. This keeps every payload section visible even when it's
   * marked unavailable.
   */
  hasData: (a: DashboardTelemetry) => boolean;
  /**
   * Lazy loader. We `lazy()` once at the module level and reuse the same
   * exotic component across renders so the chunk only loads once.
   */
  Component: LazyExoticComponent<ComponentType<TabComponentProps>>;
  /**
   * Method disclosure rendered at the bottom of each tab. Pulled from the
   * shared moduleRegistry by id where possible so /modules and the tab show
   * the same copy.
   */
  howItWorks: HowThisWorks;
}

/**
 * Builds a `HowThisWorks` payload from an entry in the moduleRegistry.
 * Falls back to a sensible default when the module is missing.
 */
function methodFrom(id: string, fallbackBody: string): HowThisWorks {
  const m = moduleById(id);
  if (!m) return { body: fallbackBody };
  return {
    title: `How ${m.name} works`,
    body: m.how,
    algorithms: m.algorithms,
    source: m.source,
  };
}

export const REPORT_TABS: TabDef[] = [
  {
    id: "overview",
    label: "Overview",
    Icon: Gauge,
    hasData: () => true,
    Component: lazy(() => import("./OverviewTab")),
    howItWorks: {
      title: "How Overview is built",
      body: "Top-line metrics, signals, anomaly snapshot, distribution, health components, and pipeline timings — assembled from the cached /api/analyze payload. Every panel can render with missing fields by falling back to an empty state.",
      algorithms: ["safe accessors", "default-shape factories"],
      source: "modules/frontend_api.py + modules/health_score.py",
    },
  },
  {
    id: "profile",
    label: "Profile",
    Icon: TableIcon,
    hasData: (a) => isPresent(a.profile),
    Component: lazy(() => import("./ProfileTab")),
    howItWorks: methodFrom("profile", "Pandas describe + dtype inference + null rates."),
  },
  {
    id: "statistics",
    label: "Statistics",
    Icon: Sigma,
    hasData: (a) => isPresent(a.deepStatisticsV2) || isPresent(a.deepStatistics),
    Component: lazy(() => import("./StatisticsTab")),
    howItWorks: methodFrom("statistics", "Per-column normality battery and bivariate effect sizes."),
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
    howItWorks: methodFrom("anomalies", "Seven-detector ensemble averaged into a single agreement score."),
  },
  {
    id: "ml-lab",
    label: "ML Lab",
    Icon: Brain,
    hasData: (a) =>
      Boolean(a.modelLeaderboard?.available) || isPresent(a.targetAnalysis),
    Component: lazy(() => import("./MlLabTab")),
    howItWorks: methodFrom("ml-lab", "5-fold CV across baseline + tree + boosted + linear families."),
  },
  {
    id: "shap",
    label: "SHAP",
    Icon: LineChart,
    hasData: (a) => Boolean(a.explainability?.available),
    Component: lazy(() => import("./ShapTab")),
    howItWorks: methodFrom("shap", "TreeSHAP for tree models with permutation fallback."),
  },
  {
    id: "timeseries",
    label: "Time-series",
    Icon: Activity,
    hasData: (a) => Boolean(a.timeSeries?.available),
    Component: lazy(() => import("./TimeSeriesTab")),
    howItWorks: methodFrom(
      "timeseries",
      "STL decomposition + ADF/KPSS stationarity + Holt-Winters forecast.",
    ),
  },
  {
    id: "text",
    label: "Text",
    Icon: FileText,
    hasData: (a) => Boolean(a.textProfile?.available),
    Component: lazy(() => import("./TextTab")),
    howItWorks: methodFrom("text", "TF-IDF + LSA over detected text columns."),
  },
  {
    id: "drift",
    label: "Drift",
    Icon: Waves,
    hasData: () => true,
    Component: lazy(() => import("./DriftTab")),
    howItWorks: methodFrom(
      "drift",
      "Per-column KS / chi-square / PSI between two datasets.",
    ),
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
    howItWorks: methodFrom(
      "diagnostics",
      "Holdout residuals, VIF, and target-leakage heuristics.",
    ),
  },
  {
    id: "segments",
    label: "Segments",
    Icon: LayoutGrid,
    hasData: (a) => isPresent(a.segmentAnalysis),
    Component: lazy(() => import("./SegmentsTab")),
    howItWorks: methodFrom(
      "segments",
      "Group-by slices over interpretable categorical or binned numeric columns.",
    ),
  },
  {
    id: "cleaning",
    label: "Cleaning",
    Icon: Sparkles,
    hasData: (a) => isPresent(a.cleaning),
    Component: lazy(() => import("./CleaningTab")),
    howItWorks: methodFrom(
      "cleaning",
      "Imputation, dedupe, outlier clipping, and a health rescore on the cleaned dataset.",
    ),
  },
  {
    id: "charts",
    label: "Charts",
    Icon: ImageIcon,
    hasData: (a) =>
      safeArr(a.charts).length > 0 || Boolean(a.explainability?.summary_chart_path),
    Component: lazy(() => import("./ChartsTab")),
    howItWorks: {
      title: "How charts are produced",
      body: "Backend modules write PNGs to /static/charts/ as side-effects of the pipeline. The chart planner picks which figures are worth generating per dataset.",
      algorithms: ["seaborn", "matplotlib"],
      source: "modules/chart_planner.py",
    },
  },
  {
    id: "ai-report",
    label: "AI Report",
    Icon: BookOpen,
    // The AI report is now generated on demand. Treat the tab as always
    // having data so the user can click into it and generate when ready.
    hasData: () => true,
    Component: lazy(() => import("./AiReportTab")),
    howItWorks: methodFrom(
      "ai-report",
      "LLM agent loop with structured output validation and graceful fallback.",
    ),
  },
  {
    id: "data-preview",
    label: "Data Preview",
    Icon: Eye,
    hasData: (a) => safeArr((a.profile as any)?.preview).length > 0,
    Component: lazy(() => import("./DataPreviewTab")),
    howItWorks: {
      title: "How Data Preview is built",
      body: "Renders the first 15 rows from `profile.preview`. The backend caps the preview to keep payloads small.",
      algorithms: ["pandas head"],
      source: "modules/profiler.py",
    },
  },
  {
    id: "timings",
    label: "Timings",
    Icon: Timer,
    hasData: (a) => isPresent((a as any).timings_ms),
    Component: lazy(() => import("./TimingsTab")),
    howItWorks: {
      title: "How timings are measured",
      body: "Each pipeline phase wraps itself in `perf_counter`; the orchestrator reports per-phase milliseconds plus the total.",
      algorithms: ["perf_counter"],
      source: "modules/pipeline.py",
    },
  },
  {
    id: "ask",
    label: "Ask Anything",
    Icon: MessageSquare,
    hasData: () => true,
    Component: lazy(() => import("./AskAnythingTab")),
    howItWorks: methodFrom(
      "ask",
      "Agentic Q&A loop scoped to the cached analysis result. Tools are read-only.",
    ),
  },
];

export const TAB_IDS: string[] = REPORT_TABS.map((t) => t.id);

export function findTab(id: string | undefined): TabDef {
  if (!id) return REPORT_TABS[0];
  return REPORT_TABS.find((t) => t.id === id) ?? REPORT_TABS[0];
}
