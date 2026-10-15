import { motion } from "framer-motion";
import { DeepStatsV2Panel } from "@/components/dashboard/DeepStatsV2Panel";
import { EmptyState } from "@/components/ui/EmptyState";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent } from "@/lib/safe";
import type { TabComponentProps } from "./tabRegistry";

/** Statistics tab — renders the canonical bounded statistics result. */
export default function StatisticsTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const v2 = t.deepStatisticsV2;

  if (isPresent(v2)) {
    return (
      <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
        <motion.div variants={staggerChild}>
          <DeepStatsV2Panel deepStats={v2 as Parameters<typeof DeepStatsV2Panel>[0]["deepStats"]} />
        </motion.div>
      </motion.div>
    );
  }

  return (
    <EmptyState
      title="Statistics not available for this dataset"
      hint="Run Prism at Standard depth or above to populate this view."
    />
  );
}
