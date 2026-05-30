import { useEffect, useState } from "react";
import { SiteShell, type Route } from "@/components/site/SiteShell";
import { useHashRoute } from "@/router/HashRouter";
import { HomePage } from "@/pages/HomePage";
import { AnalyzePage } from "@/pages/AnalyzePage";
import { ReportPage } from "@/pages/ReportPage";
import { ChartsPage } from "@/pages/ChartsPage";
import { ModulesPage } from "@/pages/ModulesPage";
import { AboutPage } from "@/pages/AboutPage";
import type { DashboardTelemetry } from "@/data/mockTelemetry";
import { getAnalysis, setAnalysis, subscribeAnalysis } from "@/lib/analysisStore";

export default function App() {
  const { route, navigate } = useHashRoute();
  const [telemetry, setLocal] = useState<DashboardTelemetry | null>(() => getAnalysis());

  // Keep local state in lock-step with the module-level cache so any page
  // that calls `setAnalysis()` propagates here without prop drilling.
  useEffect(() => subscribeAnalysis((next) => setLocal(next)), []);

  // SiteShell expects a `(Route) => void` callback. The hash router's
  // `navigate` takes an optional second arg for tabs, which is harmless here.
  const onNavigate = (next: Route) => navigate(next);

  return (
    <SiteShell route={route} onNavigate={onNavigate}>
      {route === "home" ? <HomePage onNavigate={onNavigate} /> : null}
      {route === "analyze" ? (
        <AnalyzePage
          onNavigate={onNavigate}
          onResult={(r) => {
            setAnalysis(r);
          }}
        />
      ) : null}
      {route === "report" ? <ReportPage telemetry={telemetry} onNavigate={onNavigate} /> : null}
      {route === "charts" ? <ChartsPage telemetry={telemetry} onNavigate={onNavigate} /> : null}
      {route === "modules" ? <ModulesPage /> : null}
      {route === "about" ? <AboutPage /> : null}
    </SiteShell>
  );
}
