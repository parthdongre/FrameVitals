import { motion } from "framer-motion";
import { TimeSeriesPanel } from "@/components/dashboard/TimeSeriesPanel";
import { EmptyState } from "@/components/ui/EmptyState";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import type { TabComponentProps } from "./tabRegistry";

export default function TimeSeriesTab({ analysis }: TabComponentProps) {
  const ts = (analysis as unknown as { timeSeries?: { available?: boolean } }).timeSeries;

  if (!ts || !ts.available) {
    return (
      <EmptyState
        title="Time-series not available for this dataset"
        hint="No confidently date-like column was detected. The module requires at least one parseable date column."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
      <motion.div variants={staggerChild}>
        <TimeSeriesPanel timeSeries={ts as any} />
      </motion.div>
    </motion.div>
  );
}
