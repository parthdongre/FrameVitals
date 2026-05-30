import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { ModelLeaderboardPanel } from "@/components/dashboard/ModelLeaderboardPanel";
import { LeaderboardBars } from "@/charts/LeaderboardBars";
import { FeatureImportanceBars } from "@/charts/FeatureImportanceBars";
import { EmptyState } from "@/components/ui/EmptyState";
import { KeyValueGrid, type KeyValueItem } from "@/components/ui/KeyValueGrid";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeArr, safeNum, safeObj, safeStr } from "@/lib/safe";
import { formatLabel, formatNumber, formatPercent } from "@/lib/format";
import type { TabComponentProps } from "./tabRegistry";

/**
 * ML Lab — model leaderboard panel + leaderboard bar chart + target analysis
 * key/value summary + baseline model CV scores + feature importance bars.
 */
export default function MlLabTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const leaderboard = t.modelLeaderboard;
  const targetAnalysis = safeObj(t.targetAnalysis, {} as Record<string, unknown>);
  const featureImportance = safeObj(t.featureImportance, {} as Record<string, unknown>);
  const baseline = safeObj(t.baselineModel, {} as Record<string, unknown>);
  const target = safeStr(t.selectedTargetColumn, safeStr(targetAnalysis.target_column, ""));
  const candidates = safeArr<string>(t.targetCandidates);

  const lbAvailable = Boolean((leaderboard as { available?: boolean })?.available);
  const targetAvailable = isPresent(targetAnalysis) && targetAnalysis.available !== false;
  const fiAvailable = isPresent(featureImportance) && featureImportance.available !== false;
  const baselineAvailable = isPresent(baseline) && baseline.available !== false;

  if (!lbAvailable && !targetAvailable && !fiAvailable && !baselineAvailable) {
    return (
      <EmptyState
        title="ML Lab not available for this dataset"
        hint={
          target
            ? "The pipeline did not produce ML outputs for this target column."
            : "Pick a target column on the Analyze page to unlock the ML lab."
        }
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-10">
      <motion.section variants={staggerChild} className="grid gap-3 sm:grid-cols-3">
        <span className="rounded-md border border-line bg-bg-1 px-3 py-2 font-mono text-[11px] text-ink-3">
          <span className="uppercase tracking-[0.2em] text-ink-4">Target · </span>
          <span className="text-ink-1">{target || "(none)"}</span>
        </span>
        {targetAnalysis.task_type ? (
          <span className="rounded-md border border-line bg-bg-1 px-3 py-2 font-mono text-[11px] text-ink-3">
            <span className="uppercase tracking-[0.2em] text-ink-4">Task · </span>
            <span className="text-ink-1">{safeStr(targetAnalysis.task_type, "—")}</span>
          </span>
        ) : null}
        {candidates.length ? (
          <span className="rounded-md border border-line bg-bg-1 px-3 py-2 font-mono text-[11px] text-ink-3">
            <span className="uppercase tracking-[0.2em] text-ink-4">Candidates · </span>
            <span className="text-ink-1">{candidates.slice(0, 4).join(", ")}{candidates.length > 4 ? "…" : ""}</span>
          </span>
        ) : null}
      </motion.section>

      {lbAvailable ? (
        <>
          <motion.div variants={staggerChild}>
            <ModelLeaderboardPanel leaderboard={leaderboard as any} />
          </motion.div>
          <motion.div variants={staggerChild}>
            <LeaderboardBars leaderboard={leaderboard} />
          </motion.div>
        </>
      ) : (
        <motion.div variants={staggerChild}>
          <EmptyState
            compact
            title="No leaderboard for this run"
            hint="Pick a target column and re-run in standard or deeper mode."
          />
        </motion.div>
      )}

      {targetAvailable ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Target analysis</Eyebrow>
          <KeyValueGrid items={kvFromTargetAnalysis(targetAnalysis)} cols={2} />
        </motion.section>
      ) : null}

      {baselineAvailable ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Baseline model</Eyebrow>
          <KeyValueGrid items={kvFromBaseline(baseline)} cols={2} />
        </motion.section>
      ) : null}

      {fiAvailable ? (
        <motion.div variants={staggerChild}>
          <FeatureImportanceBars
            importance={featureImportance}
            eyebrow="Feature importance"
            title="Feature importance"
            description="Mean absolute importance per feature."
          />
        </motion.div>
      ) : null}
    </motion.div>
  );
}

/* --- helpers ---------------------------------------------------------- */

function kvFromTargetAnalysis(ta: Record<string, unknown>): KeyValueItem[] {
  const items: KeyValueItem[] = [];

  const push = (label: string, value: KeyValueItem["value"], mono = true) => {
    if (value !== undefined && value !== null && value !== "" && value !== "—") {
      items.push({ label, value, mono });
    }
  };

  push("Task type", safeStr(ta.task_type, "—"));
  push("Target column", safeStr(ta.target_column, "—"));
  push(
    "Class count",
    typeof ta.n_classes === "number" ? ta.n_classes : undefined,
  );

  const dist = safeObj(ta.class_distribution, {} as Record<string, unknown>);
  if (Object.keys(dist).length) {
    const summary = Object.entries(dist)
      .slice(0, 4)
      .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toLocaleString() : String(v)}`)
      .join(" · ");
    push("Class distribution", summary, false);
  }

  if (typeof ta.imbalance_ratio === "number") {
    push("Imbalance ratio", formatNumber(ta.imbalance_ratio, 2));
  }
  if (typeof ta.skew === "number") push("Skew", formatNumber(ta.skew, 3));
  if (typeof ta.missing_rate === "number") push("Missing rate", formatPercent(ta.missing_rate));
  if (typeof ta.unique_count === "number") push("Unique values", ta.unique_count.toLocaleString());

  // Catch-all for any other primitive metrics.
  for (const [k, v] of Object.entries(ta)) {
    if (
      ["task_type", "target_column", "n_classes", "class_distribution", "imbalance_ratio",
        "skew", "missing_rate", "unique_count", "available", "message"].includes(k)
    ) continue;
    if (typeof v === "number" && Number.isFinite(v)) {
      push(formatLabel(k), formatNumber(v, 3));
    }
  }

  return items;
}

function kvFromBaseline(b: Record<string, unknown>): KeyValueItem[] {
  const items: KeyValueItem[] = [];

  const cv = safeObj(b.cv_scores, {} as Record<string, unknown>);
  for (const [metric, value] of Object.entries(cv)) {
    if (typeof value === "number" && Number.isFinite(value)) {
      items.push({ label: `CV · ${metric}`, value: formatNumber(value, 4), mono: true });
    }
  }
  for (const [k, v] of Object.entries(b)) {
    if (k === "cv_scores" || k === "available" || k === "message") continue;
    if (typeof v === "number") {
      items.push({ label: formatLabel(k), value: formatNumber(v, 4), mono: true });
    } else if (typeof v === "string") {
      items.push({ label: formatLabel(k), value: v });
    }
  }
  return items;
}
