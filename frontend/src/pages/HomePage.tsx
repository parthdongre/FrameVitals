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
    title: "Prism",
    body:
      "Turn a dataset into one coherent view of its condition, structure, risks, and modelling potential.",
  },
  {
    eyebrow: "02",
    title: "Axiom",
    body:
      "Establish expectations from trusted data and test whether new data still respects them.",
  },
  {
    eyebrow: "03",
    title: "Forge",
    body:
      "Prepare a conservative transformation plan and apply it only when you are ready.",
  },
  {
    eyebrow: "04",
    title: "Tide",
    body:
      "Understand how a dataset changes between two points in time, releases, or pipeline states.",
  },
  {
    eyebrow: "05",
    title: "Pulse",
    body:
      "Capture compact health states that can be retained and compared as the dataset evolves.",
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
          <Eyebrow>FrameVitals protocols</Eyebrow>
          <PageTitle subtitle="A protocol-first way to understand, trust, transform, compare, and monitor tabular data without stitching together a dozen separate workflows.">
            Read the signal in your data.
          </PageTitle>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.16 }}
            className="flex flex-wrap items-center gap-3"
          >
            <button onClick={() => onNavigate("analyze")} className="btn-primary">
              Run Prism
            </button>
            <button onClick={() => onNavigate("modules")} className="btn-ghost">
              Explore protocols →
            </button>
            <HealthChip />
          </motion.div>
        </div>
      </Section>

      <Hr label="Protocol map" />

      <Section>
        <SectionHeader
          eyebrow="FrameVitals protocol system"
          title="Five protocols. One coherent workflow."
          description="Use Prism for the broad view, then reach for the more focused protocols when your workflow calls for them."
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

  let label = "System unavailable";
  let tone: "ok" | "warn" | "off" = "off";

  if (isLoading) {
    label = "Checking system…";
    tone = "warn";
  } else if (isError || !data?.flask) {
    label = "System unavailable";
    tone = "off";
  } else {
    label = "System ready";
    tone = "ok";
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
