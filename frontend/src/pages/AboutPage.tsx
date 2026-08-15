import { Eyebrow, Hr, PageTitle, Section, SectionHeader } from "@/components/site/SiteShell";

export function AboutPage() {
  return (
    <>
      <Section className="pb-12 pt-6">
        <Eyebrow>About</Eyebrow>
        <PageTitle subtitle="FrameVitals is an installable Python package with optional web interfaces for structured tabular-data diagnostics. The analysis engine is evidence-first, while AI-assisted interpretation is an optional layer rather than a requirement.">
          What this is, and isn't.
        </PageTitle>
      </Section>

      <Hr label="Architecture" />

      <Section>
        <SectionHeader
          eyebrow="One pipeline"
          title="Multiple lenses, one orchestrator."
          description="A six-phase orchestrator runs profiling, quality scoring, deep statistics, anomaly analysis, modelling, time-series and text analysis, cleaning, visualization, and optional AI interpretation. Independent compute-heavy phases can run in parallel."
        />

        <div className="grid gap-x-12 gap-y-8 md:grid-cols-2">
          <Stat label="Core" value="Python 3.11–3.13 · pandas · NumPy" />
          <Stat label="Web" value="Flask · React · Vite · Streamlit" />
          <Stat label="Charts" value="Highcharts · seaborn · matplotlib" />
          <Stat label="ML" value="scikit-learn · optional XGBoost / LightGBM / SHAP" />
          <Stat label="Stats" value="SciPy · statsmodels" />
          <Stat label="AI" value="Optional Ollama client with deterministic fallbacks" />
        </div>
      </Section>

      <Hr label="Honesty" />

      <Section>
        <SectionHeader eyebrow="What it is" title="A structured, reproducible diagnostic report." />
        <ul className="space-y-3 text-[14px] leading-7 text-[var(--ink-2)]">
          <li>· Core diagnostics run without an LLM or cloud service.</li>
          <li>· The model leaderboard is a baseline, not a tuned production model.</li>
          <li>· Anomaly scores are signals from multiple detectors, not proof that a row is invalid.</li>
          <li>· Optional AI summaries are grounded in structured analysis output and should still be reviewed.</li>
        </ul>
      </Section>

      <Section>
        <SectionHeader eyebrow="And isn't" title="Not a replacement for expert judgment." />
        <ul className="space-y-3 text-[14px] leading-7 text-[var(--ink-2)]">
          <li>· Not a replacement for statistical sign-off on regulated decisions.</li>
          <li>· Not a forensic tool — flagged rows can be valid.</li>
          <li>· Not a black-box AutoML system — it surfaces structure and evidence for you to interpret.</li>
        </ul>
      </Section>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="eyebrow">{label}</p>
      <p className="mt-2 text-base text-[var(--ink-1)]">{value}</p>
    </div>
  );
}
