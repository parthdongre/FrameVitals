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
      "Turn structured data or an ML artifact into one coherent view of its condition, structure, risks, and useful next actions.",
  },
  {
    eyebrow: "02",
    title: "Axiom",
    body:
      "Establish expectations from a trusted source and test whether new data, graphs, tensors, or models still respect them.",
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
      "Understand how data, topology, tensors, or model parameters change between states.",
  },
  {
    eyebrow: "05",
    title: "Pulse",
    body:
      "Capture compact health states that can be retained as data, models, and training systems evolve.",
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
          <PageTitle subtitle="A protocol-first way to diagnose structured data and ML systems without stitching together separate profiling, graph, tensor, checkpoint, validation, and monitoring workflows.">
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
          description="Use Prism for the broad view, then Axiom, Forge, Tide, and Pulse for trust, transformation, change, and monitoring. The Python API recognizes tables, graphs, tensors, nested data, relational projects, and model artifacts."
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
          title="Diagnostics are evidence, not proof."
          description="FrameVitals reports on the source you provide. It does not replace domain expertise, statistical sign-off on regulated decisions, or production model validation."
        />

        <ul className="grid gap-3 text-[14px] leading-7 text-[var(--ink-2)] sm:grid-cols-2">
          <li>· FrameVitals describes the source you provide; it cannot supply missing domain context.</li>
          <li>· Predictive signals are analytical baselines, not production approval or deployment guidance.</li>
          <li>· Flagged rows are signals worth reviewing; they are not proof that an observation is invalid.</li>
          <li>· Optional narratives are interpretation aids, not authorities.</li>
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
