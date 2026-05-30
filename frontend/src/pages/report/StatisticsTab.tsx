import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { DeepStatsV2Panel } from "@/components/dashboard/DeepStatsV2Panel";
import { EmptyState } from "@/components/ui/EmptyState";
import { KeyValueGrid } from "@/components/ui/KeyValueGrid";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeObj } from "@/lib/safe";
import { formatLabel, formatNumber } from "@/lib/format";
import type { TabComponentProps } from "./tabRegistry";

/**
 * Statistics tab — Deep Statistics v2 with a fallback to the legacy
 * `deepStatistics` summary when v2 is missing.
 */
export default function StatisticsTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const v2 = t.deepStatisticsV2;
  const legacy = t.deepStatistics;

  if (isPresent(v2)) {
    return (
      <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
        <motion.div variants={staggerChild}>
          <DeepStatsV2Panel deepStats={v2 as Parameters<typeof DeepStatsV2Panel>[0]["deepStats"]} />
        </motion.div>
      </motion.div>
    );
  }

  if (isPresent(legacy)) {
    return (
      <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Deep statistics (legacy)</Eyebrow>
          <p className="mb-4 text-[13px] leading-6 text-ink-3">
            Deep Statistics v2 was not available for this dataset; falling back to the legacy
            summary the pipeline emitted.
          </p>
          <KeyValueGrid
            items={Object.entries(safeObj(legacy, {} as Record<string, unknown>))
              .slice(0, 16)
              .map(([k, v]) => ({
                label: formatLabel(k),
                value:
                  typeof v === "number"
                    ? formatNumber(v)
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
        </motion.section>
      </motion.div>
    );
  }

  return (
    <EmptyState
      title="Statistics not available for this dataset"
      hint="Run the analyzer in standard or deeper mode to populate the deep-statistics panel."
    />
  );
}
