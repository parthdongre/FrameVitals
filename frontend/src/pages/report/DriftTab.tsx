import { motion } from "framer-motion";
import { DriftPanel } from "@/components/dashboard/DriftPanel";
import { Eyebrow } from "@/components/site/SiteShell";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import type { TabComponentProps } from "./tabRegistry";

/**
 * Drift tab — embeds the existing DriftPanel which already owns its own
 * upload state and the two-files / split-by-date toggle. The tab adds an
 * editorial intro on top so the section's purpose is clear without scrolling
 * into the panel's own description.
 */
export default function DriftTab(_props: TabComponentProps) {
  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-6">
      <motion.section variants={staggerChild}>
        <Eyebrow className="mb-3">Distribution shift</Eyebrow>
        <p className="max-w-3xl text-[14px] leading-7 text-ink-2">
          Compare two datasets — or split one chronologically — and quantify how much each column
          has shifted. Severity is bucketed into stable / minor / moderate / severe via PSI, with
          KS and chi-square supporting evidence. Click a row in the table to overlay the reference
          and current distributions.
        </p>
      </motion.section>

      <motion.div variants={staggerChild}>
        <DriftPanel />
      </motion.div>
    </motion.div>
  );
}
