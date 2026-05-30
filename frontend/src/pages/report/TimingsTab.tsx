import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeNum, safeObj } from "@/lib/safe";
import { formatLabel, formatMs } from "@/lib/format";
import type { TabComponentProps } from "./tabRegistry";

interface TimingRow extends Record<string, unknown> {
  phase: string;
  ms: number;
  share: number;
}

/**
 * Timings tab — per-phase milliseconds sorted descending, with a small share
 * column so users see which step dominated the run.
 */
export default function TimingsTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const timings = safeObj(t.timings_ms, {} as Record<string, unknown>);
  const totalDuration = safeNum(t.analysisDurationMs, 0);

  const rows: TimingRow[] = Object.entries(timings)
    .filter(([k]) => !k.startsWith("phase3_tasks"))
    .map(([phase, raw]) => ({ phase, ms: safeNum(raw, 0), share: 0 }))
    .filter((r) => Number.isFinite(r.ms));

  const measuredTotal = rows.reduce((acc, r) => acc + r.ms, 0);
  const denom = totalDuration > 0 ? totalDuration : measuredTotal;
  for (const row of rows) {
    row.share = denom > 0 ? row.ms / denom : 0;
  }
  rows.sort((a, b) => b.ms - a.ms);

  if (!rows.length && !isPresent(timings)) {
    return (
      <EmptyState
        title="No timings recorded"
        hint="The orchestrator did not emit per-phase timings for this run."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-8">
      <motion.section
        variants={staggerChild}
        className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-3"
      >
        <Stat
          label="Total"
          value={totalDuration > 0 ? formatMs(totalDuration) : "—"}
          sub="end-to-end pipeline duration"
        />
        <Stat
          label="Measured"
          value={measuredTotal > 0 ? formatMs(measuredTotal) : "—"}
          sub="sum of per-phase timings"
        />
        <Stat label="Phases" value={String(rows.length)} sub="reported by orchestrator" />
      </motion.section>

      <motion.section variants={staggerChild}>
        <Eyebrow className="mb-3">Per-phase</Eyebrow>
        <DataTable
          columns={timingColumns}
          rows={rows}
          rowKey={(r) => r.phase}
          maxHeight="60vh"
        />
      </motion.section>
    </motion.div>
  );
}

const timingColumns: DataTableColumn<TimingRow>[] = [
  { key: "phase", header: "Phase", cell: (r) => formatLabel(r.phase), wrap: true },
  {
    key: "ms",
    header: "ms",
    mono: true,
    align: "right",
    cell: (r) => formatMs(r.ms),
  },
  {
    key: "share",
    header: "% share",
    mono: true,
    align: "right",
    cell: (r) => `${(r.share * 100).toFixed(1)}%`,
  },
  {
    key: "bar",
    header: "",
    width: "180px",
    cell: (r) => <ShareBar share={r.share} />,
  },
];

function ShareBar({ share }: { share: number }) {
  const pct = Math.max(0, Math.min(1, share));
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-bg-2">
      <span
        className="absolute inset-y-0 left-0 bg-accent"
        style={{ width: `${pct * 100}%` }}
        aria-hidden="true"
      />
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="bg-bg-1 p-5">
      <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">{label}</p>
      <p className="mt-2 text-[22px] font-semibold tabular-nums text-ink-1">{value}</p>
      <p className="mt-1 text-[11px] uppercase tracking-[0.24em] text-ink-3">{sub}</p>
    </div>
  );
}
