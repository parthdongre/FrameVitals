import { lazy, Suspense, useEffect, useState } from "react";
import { SiteShell, type Route } from "@/components/site/SiteShell";
import { useHashRoute } from "@/router/HashRouter";
const HomePage = lazy(() =>
  import("@/pages/HomePage").then((m) => ({ default: m.HomePage })),
);
const AnalyzePage = lazy(() =>
  import("@/pages/AnalyzePage").then((m) => ({ default: m.AnalyzePage })),
);
const ReportPage = lazy(() =>
  import("@/pages/ReportPage").then((m) => ({ default: m.ReportPage })),
);
const ChartsPage = lazy(() =>
  import("@/pages/ChartsPage").then((m) => ({ default: m.ChartsPage })),
);
const ModulesPage = lazy(() =>
  import("@/pages/ModulesPage").then((m) => ({ default: m.ModulesPage })),
);
const AboutPage = lazy(() =>
  import("@/pages/AboutPage").then((m) => ({ default: m.AboutPage })),
);
import type { DashboardTelemetry } from "@/data/telemetry";
import { getAnalysis, subscribeAnalysis } from "@/lib/analysisStore";

export default function App() {
  const { route, navigate } = useHashRoute();
  const [telemetry, setLocal] = useState<DashboardTelemetry | null>(() =>
    getAnalysis(),
  );

  // Keep local state in lock-step with the module-level cache so any page
  // that calls `setAnalysis()` propagates here without prop drilling.
  useEffect(() => subscribeAnalysis((next) => setLocal(next)), []);

  // SiteShell expects a `(Route) => void` callback. The hash router's
  // `navigate` takes an optional second arg for tabs, which is harmless here.
  const onNavigate = (next: Route) => navigate(next);

  return (
    <SiteShell route={route} onNavigate={onNavigate}>
      <Suspense fallback={<p role="status">Loading workspace…</p>}>
        {route === "home" ? <HomePage onNavigate={onNavigate} /> : null}
        {route === "analyze" ? <AnalyzePage onNavigate={onNavigate} /> : null}
        {route === "report" ? (
          <ReportPage telemetry={telemetry} onNavigate={onNavigate} />
        ) : null}
        {route === "charts" ? (
          <ChartsPage telemetry={telemetry} onNavigate={onNavigate} />
        ) : null}
        {route === "modules" ? <ModulesPage /> : null}
        {route === "about" ? <AboutPage /> : null}
      </Suspense>
    </SiteShell>
  );
}
