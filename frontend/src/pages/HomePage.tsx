import { motion } from "framer-motion";
import { Eyebrow, Hr, PageTitle, Section, SectionHeader } from "@/components/site/SiteShell";
import type { Route } from "@/components/site/SiteShell";
import { Parallax } from "@/components/site/Parallax";
import { useHealthQuery } from "@/hooks/useHealthQuery";
import { cn } from "@/lib/utils";

interface HomePageProps {
  onNavigate: (route: Route) => void;
}

const FEATURES: { eyebrow: string; title: string; body: string }[] = [
  {
    eyebrow: "01",
    title: "Statistical depth",
    body:
      "Per-column normality tests, distribution diagnostics, bootstrap confidence intervals, and bivariate tests surface structure that basic profiling misses.",
  },
  {
    eyebrow: "02",
    title: "Anomaly ensemble",
    body:
      "Core detectors include IsolationForest, LOF, EllipticEnvelope, robust z-score, and Mahalanobis distance. Optional PyOD support adds ECOD and COPOD before scores are normalized into one ensemble view.",
  },
  {
    eyebrow: "03",
    title: "Model leaderboard + explainability",
    body:
      "Target-aware baseline models are cross-validated and compared consistently. Optional XGBoost and LightGBM expand the leaderboard, while SHAP is used when available with deterministic feature-importance fallbacks otherwise.",
  },
  {
    eyebrow: "04",
    title: "Time-series and text",
    body:
      "Date-like columns can be inspected for temporal structure and stationarity, while free-text columns receive dedicated profiling so they are not treated as ordinary categories.",
  },
  {
    eyebrow: "05",
    title: "Drift / compare mode",
    body:
      "Compare two datasets — or split one chronologically — and quantify column-by-column shift with PSI, Kolmogorov-Smirnov, and chi-square diagnostics.",
  },
  {
    eyebrow: "06",
    title: "Optional AI interpretation",
    body:
      "AI-assisted summaries can use a local Ollama service when configured. Core diagnostics do not require an LLM, and deterministic structured output remains available when AI integrations are absent.",
  },
];

const item = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0 },
};
const list = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05, delayChildren: 0.1 } },
};

export function HomePage({ onNavigate }: HomePageProps) {
  return (
    <>
      <Section className="relative pb-12 pt-6">
        <Parallax
          range={42}
          className="pointer-events-none absolute right-0 top-2 -z-0 hidden select-none md:block"
        >
          <span
            aria-hidden="true"
            className="block font-mono text-[clamp(120px,12vw,200px)] font-bold leading-none text-[var(--ink-1)]/[0.04]"
          >
            FV/0.1
          </span>
        </Parallax>

        <div className="relative">
          <Eyebrow>Tabular data diagnostics</Eyebrow>
          <PageTitle subtitle="Upload a dataset and get structured, evidence-backed diagnostics for data quality, ML readiness, anomaly behavior, time-series structure, text columns, drift, modelling, and optional AI-assisted interpretation.">
            Read the signal in your data.
          </PageTitle>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="flex flex-wrap items-center gap-3"
          >
            <button onClick={() => onNavigate("analyze")} className="btn-primary">
              Analyze a dataset
            </button>
            <button onClick={() => onNavigate("modules")} className="btn-ghost">
              See what it inspects →
            </button>
            <HealthChip />
          </motion.div>
        </div>
      </Section>

      <Hr label="Inspection map" />

      <Section>
        <SectionHeader
          eyebrow="What FrameVitals inspects"
          title="Six analytical lenses."
          description="Each lens is grounded in named algorithms and structured outputs. Optional integrations extend the core engine without becoming mandatory dependencies."
        />

        <motion.div
          variants={list}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
          className="grid gap-x-12 gap-y-12 md:grid-cols-2"
        >
          {FEATURES.map((f) => (
            <motion.div
              key={f.eyebrow}
              variants={item}
              transition={{ duration: 0.55, ease: "easeOut" }}
            >
              <p className="eyebrow">{f.eyebrow}</p>
              <h3 className="mt-3 text-xl font-semibold text-[var(--ink-1)]">{f.title}</h3>
              <p className="mt-3 text-pretty text-[15px] leading-7 text-[var(--ink-2)]">{f.body}</p>
            </motion.div>
          ))}
        </motion.div>
      </Section>

      <Hr label="Important limits" />

      <Section>
        <SectionHeader
          eyebrow="Honest about the tradeoffs"
          title="This is a structured diagnostic report, not proof."
          description="FrameVitals reports on the dataset you provide. It does not replace domain expertise, statistical sign-off on regulated decisions, or production model validation."
        />

        <ul className="grid gap-3 text-[14px] leading-7 text-[var(--ink-2)] sm:grid-cols-2">
          <li>· Core analytics run locally. External AI services are only involved when you explicitly configure an optional integration.</li>
          <li>· ML metrics are baselines, not tuned production models. The leaderboard surfaces which model families respond well to the data.</li>
          <li>· Anomaly scores represent detector agreement. A flagged row can still be a valid observation.</li>
          <li>· AI-assisted summaries should be treated as interpretation aids, not authorities.</li>
        </ul>
      </Section>
    </>
  );
}

function HealthChip() {
  const { data, isLoading, isError } = useHealthQuery();

  let label = "Backend offline";
  let tone: "ok" | "warn" | "off" = "off";

  if (isLoading) {
    label = "Probing backend…";
    tone = "warn";
  } else if (isError || !data?.flask) {
    label = "Backend offline";
    tone = "off";
  } else {
    const ai = data.ollama_reachable ? "Ollama" : data.openrouter_configured ? "OpenRouter" : "no LLM";
    label = `Backend live · ${ai}`;
    tone = data.ollama_reachable || data.openrouter_configured ? "ok" : "warn";
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 font-mono text-[11px] tracking-[0.06em]",
        tone === "ok" && "border-[var(--accent-line)] bg-[var(--accent-soft)] text-[var(--accent)]",
        tone === "warn" && "border-[var(--line-strong)] bg-[var(--bg-2)] text-[var(--ink-2)]",
        tone === "off" && "border-[var(--line)] bg-[var(--bg-2)] text-[var(--ink-3)]",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          tone === "ok" && "bg-[var(--accent)]",
          tone === "warn" && "bg-[var(--warn)]",
          tone === "off" && "bg-[var(--ink-4)]",
        )}
      />
      {label}
    </span>
  );
}
