import { motion } from "framer-motion";
import { DriftPanel } from "@/components/dashboard/DriftPanel";
import { Eyebrow } from "@/components/site/SiteShell";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import type { TabComponentProps } from "./tabRegistry";

/** Tide comparison view for dataset change over time or between states. */
export default function DriftTab(_props: TabComponentProps) {
  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-6">
      <motion.section variants={staggerChild}>
        <Eyebrow className="mb-3">Tide</Eyebrow>
        <p className="max-w-3xl text-[14px] leading-7 text-ink-2">
          Compare two dataset states — or split one chronologically — and see where meaningful
          change has appeared. Open a column to compare the reference and current distributions.
        </p>
      </motion.section>

      <motion.div variants={staggerChild}>
        <DriftPanel />
      </motion.div>
    </motion.div>
  );
}
