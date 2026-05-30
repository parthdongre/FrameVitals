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
      "Per-column normality battery (Shapiro · D'Agostino · Anderson), best-fit distribution by AIC across six candidates, BCa bootstrap CIs, and bivariate tests including Mann-Whitney, Kruskal-Wallis, Cramér's V, and point-biserial.",
  },
  {
    eyebrow: "02",
    title: "Anomaly ensemble",
    body:
      "Seven detectors run in parallel — IsolationForest, LOF, EllipticEnvelope, robust z-score, Mahalanobis, ECOD, COPOD — and their normalized scores are averaged into a single ensemble for honest agreement-based flagging.",
  },
  {
    eyebrow: "03",
    title: "Model leaderboard + SHAP",
    body:
      "5-fold cross-validated leaderboard across up to 8 models including XGBoost and LightGBM, with calibrated holdout metrics. The winner gets full SHAP attribution — global ranking plus per-row stories.",
  },
  {
    eyebrow: "04",
    title: "Time-series and text",
    body:
      "Confidently date-like columns are auto-decomposed with STL, tested for stationarity (ADF + KPSS), and forecast with Holt-Winters. Free-text columns get linguistic stats, n-grams, regex pattern hits, and a TF-IDF/LSA document map.",
  },
  {
    eyebrow: "05",
    title: "Drift / compare mode",
    body:
      "Compare two datasets — or split one chronologically — and quantify column-by-column shift with PSI, Kolmogorov-Smirnov, and chi-square. Severity buckets surface the columns that actually changed.",
  },
  {
    eyebrow: "06",
    title: "LLM AI agent",
    body:
      "Planner → Executor → Critic → Writer loop running on local Ollama with Pydantic-validated structured output. Falls back gracefully to OpenRouter, then to a deterministic writer that quotes RAG-retrieved facts.",
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
        {/* Decorative parallax glyph — purely visual, doesn't carry semantics. */}
        <Parallax
          range={42}
          className="pointer-events-none absolute right-0 top-2 -z-0 hidden select-none md:block"
        >
          <span
            aria-hidden="true"
            className="block font-mono text-[clamp(120px,12vw,200px)] font-bold leading-none text-[var(--ink-1)]/[0.04]"
          >
            DL/v3
          </span>
        </Parallax>

        <div className="relative">
          <Eyebrow>LLM dataset analytics</Eyebrow>
          <PageTitle subtitle="Upload a dataset and get a structured, evidence-backed report on data quality, machine-learning readiness, anomaly behavior, time-series structure, free-text content, drift, and an LLM narrative — all in one pass.">
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
          eyebrow="What DataLens inspects"
          title="Six analytical lenses."
          description="Each lens runs independently and is grounded in named algorithms. Every section of the report ties back to one of these."
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
          title="This is a structured report, not proof."
          description="DataLens is a structured report on the dataset you upload. It does not replace domain expertise, statistical sign-off on regulated decisions, or production model validation."
        />

        <ul className="grid gap-3 text-[14px] leading-7 text-[var(--ink-2)] sm:grid-cols-2">
          <li>· All analytics run locally on your machine. Nothing is uploaded to a third party unless you explicitly configure OpenRouter for the AI agent.</li>
          <li>· ML metrics are baselines, not tuned production models. The leaderboard exists to surface which families respond well to the data.</li>
          <li>· Anomaly scores are agreement-based across seven detectors. Some structural rows surface even when valid.</li>
          <li>· The AI agent is grounded in retrieved facts and validated for structure, but should be treated as a summarizer, not an authority.</li>
        </ul>
      </Section>
    </>
  );
}

/* --------------------------------------------------------------------------
 * Health chip — small live status pill rendered next to the hero CTAs.
 * Network failures are swallowed by `useHealthQuery` (`throwOnError: false`),
 * so the worst case is a muted "offline" pip.
 * -------------------------------------------------------------------------- */
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
