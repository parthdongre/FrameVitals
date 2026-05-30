import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeArr, safeNum, safeObj, safeStr } from "@/lib/safe";
import { formatLabel, formatNumber } from "@/lib/format";
import type { TabComponentProps } from "./tabRegistry";

interface SegmentEntry extends Record<string, unknown> {
  segment_column?: string;
  analysis_type?: string;
  groups?: Array<Record<string, unknown>>;
}

/**
 * Segments tab — renders one DataTable per detected segment column. Two
 * shapes are possible from the backend:
 *
 *   - `numeric_target_by_segment`: count, mean/median/std/min/max of the target.
 *   - `segment_distribution`: count + percent share.
 */
export default function SegmentsTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const seg = safeObj(t.segmentAnalysis, {} as Record<string, unknown>);
  const available = seg.available === true;
  const results = safeArr<SegmentEntry>(seg.results);

  if (!available || !results.length) {
    return (
      <EmptyState
        title="Segments not available for this dataset"
        hint={
          isPresent(seg)
            ? safeStr(seg.message, "No suitable low-cardinality segment columns detected.")
            : "Segments require at least one low-cardinality column. Re-run with a target to compare metrics across segments."
        }
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-10">
      {results.map((entry, idx) => {
        const groups = safeArr<Record<string, unknown>>(entry.groups);
        const isNumericTarget = entry.analysis_type === "numeric_target_by_segment";
        return (
          <motion.section key={`${idx}-${entry.segment_column}`} variants={staggerChild}>
            <Eyebrow className="mb-3">
              {isNumericTarget ? "Target metrics by segment" : "Segment distribution"}
            </Eyebrow>
            <p className="mb-3 text-[14px] leading-7 text-ink-2">
              <span className="font-mono text-[12px] text-accent">
                {safeStr(entry.segment_column, "—")}
              </span>{" "}
              · {groups.length} group{groups.length === 1 ? "" : "s"}
            </p>
            <DataTable
              columns={isNumericTarget ? numericTargetColumns : distributionColumns}
              rows={groups}
              rowKey={(r, i) => `${idx}-${i}-${safeStr((r as Record<string, unknown>).group, "")}`}
              maxHeight="50vh"
            />
          </motion.section>
        );
      })}
    </motion.div>
  );
}

/* --- columns ------------------------------------------------------------ */

const numericTargetColumns: DataTableColumn<Record<string, unknown>>[] = [
  { key: "group", header: "Group", wrap: true, cell: (r) => safeStr(r.group, "—") },
  {
    key: "count",
    header: "Count",
    mono: true,
    align: "right",
    cell: (r) => safeNum(r.count, 0).toLocaleString(),
  },
  {
    key: "target_mean",
    header: "Mean",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.target_mean === "number" ? formatNumber(r.target_mean, 3) : "—",
  },
  {
    key: "target_median",
    header: "Median",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.target_median === "number" ? formatNumber(r.target_median, 3) : "—",
  },
  {
    key: "target_std",
    header: "Std",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.target_std === "number" ? formatNumber(r.target_std, 3) : "—",
  },
  {
    key: "target_min",
    header: "Min",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.target_min === "number" ? formatNumber(r.target_min, 3) : "—",
  },
  {
    key: "target_max",
    header: "Max",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.target_max === "number" ? formatNumber(r.target_max, 3) : "—",
  },
];

const distributionColumns: DataTableColumn<Record<string, unknown>>[] = [
  { key: "group", header: "Group", wrap: true, cell: (r) => safeStr(r.group, "—") },
  {
    key: "count",
    header: "Count",
    mono: true,
    align: "right",
    cell: (r) => safeNum(r.count, 0).toLocaleString(),
  },
  {
    key: "percent",
    header: "% share",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.percent === "number" ? `${r.percent.toFixed(1)}%` : "—",
  },
];

// formatLabel reserved for future use (kept import to avoid TS unused warning if needed).
void formatLabel;
