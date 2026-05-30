import { Eyebrow, Hr, PageTitle, Section, SectionHeader } from "@/components/site/SiteShell";

export function AboutPage() {
  return (
    <>
      <Section className="pb-12 pt-6">
        <Eyebrow>About</Eyebrow>
        <PageTitle subtitle="DataLens AI is a Python pipeline plus a React dashboard for structured dataset analysis. Everything is LLM, evidence-backed, and built around an honest set of tradeoffs.">
          What this is, and isn't.
        </PageTitle>
      </Section>

      <Hr label="Architecture" />

      <Section>
        <SectionHeader
          eyebrow="One pipeline"
          title="Multiple lenses, one orchestrator."
          description="A six-phase orchestrator runs profiling, quality scoring, deep statistics, anomaly ensembles, ML modelling, time-series, text, drift, and an AI summary. The phases that don't touch each other run in parallel."
        />

        <div className="grid gap-x-12 gap-y-8 md:grid-cols-2">
          <Stat label="Backend" value="Flask · Python 3.14" />
          <Stat label="Frontend" value="React 19 · Vite · Tailwind" />
          <Stat label="Charts" value="Highcharts · seaborn · matplotlib" />
          <Stat label="ML" value="scikit-learn · XGBoost · LightGBM · SHAP" />
          <Stat label="Stats" value="scipy · statsmodels · pingouin" />
          <Stat label="LLM" value="Ollama (qwen3:4b) → OpenRouter → heuristic" />
        </div>
      </Section>

      <Hr label="Honesty" />

      <Section>
        <SectionHeader eyebrow="What it is" title="A structured, reproducible report." />
        <ul className="space-y-3 text-[14px] leading-7 text-[var(--ink-2)]">
          <li>· Pipeline runs in 1–8 seconds for typical datasets thanks to parallel orchestration.</li>
          <li>· The model leaderboard is a baseline, not a tuned production model.</li>
          <li>· Anomaly scores agree across detectors — they are signal, not proof.</li>
          <li>· The AI agent is grounded in retrieved facts and validated for structure.</li>
        </ul>
      </Section>

      <Section>
        <SectionHeader eyebrow="And isn't" title="Not a replacement for expert judgment." />
        <ul className="space-y-3 text-[14px] leading-7 text-[var(--ink-2)]">
          <li>· Not a replacement for statistical sign-off on regulated decisions.</li>
          <li>· Not a forensic tool — flagged rows can be valid.</li>
          <li>· Not a black-box AutoML — it surfaces structure, you make the call.</li>
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
