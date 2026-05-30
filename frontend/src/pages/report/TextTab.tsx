import { motion } from "framer-motion";
import { TextProfilePanel } from "@/components/dashboard/TextProfilePanel";
import { EmptyState } from "@/components/ui/EmptyState";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import type { TabComponentProps } from "./tabRegistry";

export default function TextTab({ analysis }: TabComponentProps) {
  const tp = (analysis as unknown as { textProfile?: { available?: boolean } }).textProfile;

  if (!tp || !tp.available) {
    return (
      <EmptyState
        title="Text profile not available for this dataset"
        hint="No free-text columns were detected. The module requires at least one string column with sufficient vocabulary."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
      <motion.div variants={staggerChild}>
        <TextProfilePanel textProfile={tp as any} />
      </motion.div>
    </motion.div>
  );
}
