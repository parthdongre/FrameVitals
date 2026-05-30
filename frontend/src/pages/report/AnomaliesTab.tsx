import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { AnomalyEnsemblePanel } from "@/components/dashboard/AnomalyEnsemblePanel";
import { AnomalyHeatmap } from "@/charts/AnomalyHeatmap";
import { EmptyState } from "@/components/ui/EmptyState";
import { KeyValueGrid } from "@/components/ui/KeyValueGrid";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeNum, safeObj } from "@/lib/safe";
import { formatLabel, formatNumber } from "@/lib/format";
import type { AnomaliesV2 } from "@/data/payload";
import type { TabComponentProps } from "./tabRegistry";

/**
 * Anomalies tab — the full ensemble panel + an interactive heatmap of the
 * most flagged rows + the legacy `advanced.anomalies` summary block when
 * present.
 */
export default function AnomaliesTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const v2 = t.anomaliesV2 as AnomaliesV2 | undefined;
  const advanced = safeObj(t.advanced, {} as Record<string, unknown>);
  const summary = safeObj(advanced.anomalies, {} as Record<string, unknown>);

  const hasEnsemble = Boolean(v2?.available);
  const hasAdvanced = isPresent(summary);

  if (!hasEnsemble && !hasAdvanced) {
    return (
      <EmptyState
        title="Anomalies not available for this dataset"
        hint="Run the analyzer in standard mode or higher to populate the anomaly ensemble."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
      {hasEnsemble ? (
        <>
          <motion.div variants={staggerChild}>
            <AnomalyEnsemblePanel anomalies={v2 as any} />
          </motion.div>
          <motion.div variants={staggerChild}>
            <AnomalyHeatmap anomalies={v2 ?? null} />
          </motion.div>
        </>
      ) : null}

      {hasAdvanced ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Advanced anomaly snapshot</Eyebrow>
          <KeyValueGrid
            items={Object.entries(summary)
              .slice(0, 10)
              .map(([k, v]) => ({
                label: formatLabel(k),
                value:
                  typeof v === "number"
                    ? formatNumber(v, 3)
                    : typeof v === "string"
                      ? v
                      : Array.isArray(v)
                        ? `${v.length} entries`
                        : typeof v === "object" && v !== null
                          ? `${Object.keys(v as object).length} entries`
                          : String(v ?? "—"),
                mono: typeof v === "number",
              }))}
            cols={2}
            compact
          />
          {safeNum((summary as { count?: number }).count, 0) > 0 ? (
            <p className="mt-3 font-mono text-[11px] text-ink-3">
              {safeNum((summary as { count?: number }).count, 0)} flagged from the advanced detector run.
            </p>
          ) : null}
        </motion.section>
      ) : null}
    </motion.div>
  );
}
