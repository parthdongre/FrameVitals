import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { ExplainabilityPanel } from "@/components/dashboard/ExplainabilityPanel";
import { ShapGlobalBars } from "@/charts/ShapGlobalBars";
import { EmptyState } from "@/components/ui/EmptyState";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import type { TabComponentProps } from "./tabRegistry";
import type { Explainability } from "@/data/payload";

function getBackendBaseUrl(): string {
  const env = (import.meta as any).env ?? {};
  const configured = (env.VITE_BACKEND_URL ?? "").toString().trim();
  if (configured) return configured.replace(/\/$/, "");
  if (typeof window !== "undefined" && window.location.port === "5173") {
    return "http://127.0.0.1:5055";
  }
  return "";
}

/**
 * SHAP tab — interactive global importance bars + the existing
 * ExplainabilityPanel (per-row stories, summary chart, errors).
 */
export default function ShapTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as { explainability?: Explainability };
  const exp = t.explainability;

  if (!exp || !exp.available) {
    return (
      <EmptyState
        title="SHAP not available for this dataset"
        hint="The explainability module runs once the leaderboard has a winner. Re-run with a target column to populate it."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
      <motion.section variants={staggerChild} className="grid gap-2 sm:grid-cols-3">
        <span className="rounded-md border border-line bg-bg-1 px-3 py-2 font-mono text-[11px] text-ink-3">
          <span className="uppercase tracking-[0.2em] text-ink-4">Method · </span>
          <span className="text-ink-1">{exp.method ?? "—"}</span>
        </span>
        <span className="rounded-md border border-line bg-bg-1 px-3 py-2 font-mono text-[11px] text-ink-3">
          <span className="uppercase tracking-[0.2em] text-ink-4">Model · </span>
          <span className="text-ink-1">{exp.model ?? "—"}</span>
        </span>
        <span className="rounded-md border border-line bg-bg-1 px-3 py-2 font-mono text-[11px] text-ink-3">
          <span className="uppercase tracking-[0.2em] text-ink-4">Test rows · </span>
          <span className="text-ink-1 tabular-nums">{exp.n_test_rows_explained ?? 0}</span>
        </span>
      </motion.section>

      <motion.div variants={staggerChild}>
        <ShapGlobalBars explainability={exp} />
      </motion.div>

      <motion.div variants={staggerChild}>
        <ExplainabilityPanel
          explainability={exp as any}
          backendBaseUrl={getBackendBaseUrl()}
        />
      </motion.div>

      {Array.isArray(exp.errors) && exp.errors.length > 0 ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Errors during explanation</Eyebrow>
          <ul className="space-y-1 rounded-md border border-line bg-bg-1 p-4 font-mono text-[12px] leading-6 text-ink-3">
            {exp.errors.map((e, i) => (
              <li key={i} className="text-rose-300">· {e}</li>
            ))}
          </ul>
        </motion.section>
      ) : null}
    </motion.div>
  );
}
