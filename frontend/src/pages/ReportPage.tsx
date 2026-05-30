import { Suspense, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Eyebrow, PageTitle, Section } from "@/components/site/SiteShell";
import type { Route } from "@/components/site/SiteShell";
import { useHashRoute } from "@/router/HashRouter";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { EmptyState } from "@/components/ui/EmptyState";
import { Tabs } from "@/components/ui/Tabs";
import { HowThisWorks } from "@/components/site/HowThisWorks";
import { findTab, REPORT_TABS } from "./report/tabRegistry";
import { tabVariants } from "@/components/site/Variants";
import { setAnalysis as cacheAnalysis, getAnalysis, subscribeAnalysis } from "@/lib/analysisStore";
import { safeStr } from "@/lib/safe";
import type { DashboardTelemetry } from "@/data/payload";

interface ReportPageProps {
  telemetry: DashboardTelemetry | null;
  onNavigate: (r: Route) => void;
}

function getBackendBaseUrl(): string {
  const env = (import.meta as any).env ?? {};
  const configured = (env.VITE_BACKEND_URL ?? "").toString().trim();
  if (configured) return configured.replace(/\/$/, "");
  if (typeof window !== "undefined" && window.location.port === "5173") {
    return "http://127.0.0.1:5055";
  }
  return "";
}

export function ReportPage({ telemetry, onNavigate }: ReportPageProps) {
  // Bridge between the legacy prop-based handoff (App.tsx still passes
  // telemetry through props) and the module-level store. Whichever lands
  // first wins; subsequent updates from either side stay in sync.
  const [analysis, setLocal] = useState<DashboardTelemetry | null>(
    () => telemetry ?? getAnalysis(),
  );

  useEffect(() => {
    if (telemetry) {
      cacheAnalysis(telemetry);
      setLocal(telemetry);
    }
  }, [telemetry]);

  useEffect(() => {
    return subscribeAnalysis((next) => setLocal(next));
  }, []);

  // No analysis yet — defer to the analyze CTA. Keeps the router-friendly
  // deep-link behavior: visiting `#/report` directly from a fresh tab still
  // gets a friendly redirect prompt instead of a white screen.
  // Also catches the case where the cache has a placeholder / partially-
  // failed payload (no id or no rows) — surfacing all-zeros as a confusing
  // report is worse than asking the user to re-run.
  const analysisAny = analysis as unknown as { id?: string; rows?: unknown } | null;
  const hasAnalysis =
    !!analysis &&
    typeof analysisAny?.id === "string" &&
    analysisAny.id.length > 0 &&
    typeof analysisAny?.rows === "number";

  if (!hasAnalysis) {
    return (
      <Section className="pt-6">
        <Eyebrow>Report</Eyebrow>
        <PageTitle subtitle="Run the analyzer first, then come back here for the structured report. The cache is per-session — refresh the analyze page to start a new one.">
          No analysis yet.
        </PageTitle>
        <button onClick={() => onNavigate("analyze")} className="btn-primary">
          Go to Analyze →
        </button>
      </Section>
    );
  }

  return <ReportShell analysis={analysis as DashboardTelemetry} onNavigate={onNavigate} />;
}

function ReportShell({
  analysis,
  onNavigate,
}: {
  analysis: DashboardTelemetry;
  onNavigate: (r: Route) => void;
}) {
  const { tab, navigate } = useHashRoute();
  const active = useMemo(() => findTab(tab), [tab]);
  const t = analysis as unknown as Record<string, unknown>;

  const tabItems = useMemo(
    () =>
      REPORT_TABS.map((def) => ({
        id: def.id,
        label: def.label,
        Icon: def.Icon,
        hasData: def.hasData(analysis),
      })),
    [analysis],
  );

  const filename = safeStr(t.filename, "dataset");
  const rows = typeof t.rows === "number" ? t.rows : 0;
  const cols = typeof t.columns === "number" ? t.columns : 0;
  const mode = safeStr(t.analysisMode, "");

  const downloadLinks = (t.downloadLinks ?? {}) as {
    cleaned?: string;
    report?: string;
  };
  const backendBase = getBackendBaseUrl();

  return (
    <>
      <Section className="pb-6 pt-6">
        <Eyebrow>Report</Eyebrow>
        <PageTitle
          subtitle={`${filename} · ${rows.toLocaleString()} rows × ${cols.toLocaleString()} columns${
            mode ? ` · mode ${mode.toUpperCase()}` : ""
          }`}
        >
          Read the report.
        </PageTitle>

        <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
          {downloadLinks.cleaned ? (
            <a className="btn-ghost" href={`${backendBase}${downloadLinks.cleaned}`}>
              Download cleaned CSV
            </a>
          ) : null}
          {downloadLinks.report ? (
            <a className="btn-ghost" href={`${backendBase}${downloadLinks.report}`}>
              Download PDF report
            </a>
          ) : null}
          <button onClick={() => onNavigate("analyze")} className="btn-ghost">
            Run again
          </button>
        </div>
      </Section>

      <div className="sticky top-[60px] z-20 -mx-6 border-y border-line bg-bg-0/85 px-6 py-2 backdrop-blur-md sm:-mx-10 sm:px-10 lg:-mx-16 lg:px-16">
        <Tabs
          items={tabItems}
          activeId={active.id}
          onChange={(next) => navigate("report", next)}
        />
      </div>

      <Section className="py-10">
        <AnimatePresence mode="wait">
          <motion.section
            key={active.id}
            id={`tabpanel-${active.id}`}
            role="tabpanel"
            aria-labelledby={`tab-${active.id}`}
            variants={tabVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-10"
          >
            <ErrorBoundary label={active.label}>
              <Suspense fallback={<TabSkeleton label={active.label} />}>
                {active.hasData(analysis) ? (
                  <active.Component analysis={analysis} />
                ) : (
                  <EmptyState
                    title={`${active.label} not available for this dataset`}
                    hint="The backend did not produce data for this section. Try running the analyzer in standard or deeper mode, or pick a target column."
                  />
                )}
              </Suspense>
            </ErrorBoundary>

            <HowThisWorks
              title={active.howItWorks.title}
              body={active.howItWorks.body}
              algorithms={active.howItWorks.algorithms}
              source={active.howItWorks.source}
            />
          </motion.section>
        </AnimatePresence>
      </Section>
    </>
  );
}

function TabSkeleton({ label }: { label: string }) {
  return (
    <div className="space-y-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">
        Loading {label}…
      </p>
      <div className="grid gap-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="h-12 animate-pulse rounded-md border border-line bg-bg-1"
          />
        ))}
      </div>
    </div>
  );
}
