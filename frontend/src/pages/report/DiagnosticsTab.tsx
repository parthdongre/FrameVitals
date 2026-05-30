import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { KeyValueGrid, type KeyValueItem } from "@/components/ui/KeyValueGrid";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeArr, safeNum, safeObj, safeStr } from "@/lib/safe";
import { formatLabel, formatNumber } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TabComponentProps } from "./tabRegistry";

/**
 * Diagnostics tab — model diagnostics, multicollinearity (VIF + redundant
 * groups), target leakage, plus advanced fairness/leakage flags.
 */
export default function DiagnosticsTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const md = safeObj(t.modelDiagnostics, {} as Record<string, unknown>);
  const mc = safeObj(t.multicollinearity, {} as Record<string, unknown>);
  const tl = safeObj(t.targetLeakage, {} as Record<string, unknown>);
  const advanced = safeObj(t.advanced, {} as Record<string, unknown>);
  const fairness = safeObj(advanced.fairness, {} as Record<string, unknown>);
  const advLeakage = safeObj(advanced.leakage, {} as Record<string, unknown>);

  const hasMd = md.available === true;
  const vif = safeObj(mc.vif, {} as Record<string, unknown>);
  const hasVif = vif.available === true;
  const redundant = safeObj(mc.redundant_groups, {} as Record<string, unknown>);
  const hasRedundant = redundant.available === true;
  const hasTl = tl.available === true;
  const hasFairness = isPresent(fairness);
  const hasAdvLeak = isPresent(advLeakage);

  if (!hasMd && !hasVif && !hasRedundant && !hasTl && !hasFairness && !hasAdvLeak) {
    return (
      <EmptyState
        title="Diagnostics not available for this dataset"
        hint="Diagnostics fill in once a target column is selected. Re-run with a target to populate VIF, residuals, leakage, and fairness."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-12">
      {/* ---- Model diagnostics (residuals / classification metrics) -------- */}
      {hasMd ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Model diagnostics</Eyebrow>
          <KeyValueGrid items={kvFromModelDiagnostics(md)} cols={2} />
        </motion.section>
      ) : null}

      {/* ---- Multicollinearity (VIF) ------------------------------------- */}
      {hasVif ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Multicollinearity (VIF)</Eyebrow>
          <p className="mb-3 text-[13px] leading-6 text-ink-3">
            VIF above 5 suggests moderate multicollinearity; above 10 suggests high multicollinearity.
            Status: <span className="text-ink-1">{safeStr(vif.status, "—")}</span>
          </p>
          <DataTable
            columns={vifColumns}
            rows={safeArr<VifRow>(vif.vif_scores)}
            rowKey={(r) => r.feature}
            maxHeight="60vh"
          />
        </motion.section>
      ) : null}

      {hasRedundant ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Redundant feature groups</Eyebrow>
          <ul className="space-y-2 rounded-md border border-line bg-bg-1 p-4 text-[13px] leading-6 text-ink-2">
            {safeArr<{ features?: string[]; representative?: string }>(redundant.groups).map(
              (g, i) => (
                <li key={i} className="flex flex-wrap items-baseline gap-2">
                  <span className="font-mono text-[10px] uppercase tracking-[0.28em] text-ink-3">
                    Group {i + 1}
                  </span>
                  {g.representative ? (
                    <span className="font-mono text-[12px] text-accent">
                      {g.representative}
                    </span>
                  ) : null}
                  <span className="text-ink-3">·</span>
                  <span>{safeArr<string>(g.features).join(", ") || "—"}</span>
                </li>
              ),
            )}
          </ul>
        </motion.section>
      ) : null}

      {/* ---- Target leakage --------------------------------------------- */}
      {hasTl ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Target leakage</Eyebrow>
          <p className="mb-3 text-[13px] leading-6 text-ink-3">
            Target: <span className="font-mono text-ink-1">{safeStr(tl.target_column, "—")}</span> ·
            status: <span className="text-ink-1">{safeStr(tl.status, "—")}</span>
          </p>
          <DataTable
            columns={leakageColumns}
            rows={safeArr<LeakageRow>(tl.suspect_features)}
            rowKey={(r, i) => `${i}-${r.feature ?? r.column}`}
            maxHeight="60vh"
            emptyState={
              <span className="text-ink-3">No suspicious features flagged.</span>
            }
          />
        </motion.section>
      ) : null}

      {/* ---- Advanced fairness ------------------------------------------- */}
      {hasFairness ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Fairness slices</Eyebrow>
          <KeyValueGrid items={kvFromGenericObject(fairness)} cols={2} compact />
        </motion.section>
      ) : null}

      {/* ---- Advanced leakage flags ------------------------------------- */}
      {hasAdvLeak ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Advanced leakage flags</Eyebrow>
          <KeyValueGrid items={kvFromGenericObject(advLeakage)} cols={2} compact />
        </motion.section>
      ) : null}
    </motion.div>
  );
}

/* --- helpers ---------------------------------------------------------- */

interface VifRow extends Record<string, unknown> {
  feature: string;
  vif: number | null;
  r2_explained_by_other_features?: number;
  severity?: string;
}

const vifColumns: DataTableColumn<VifRow>[] = [
  { key: "feature", header: "Feature", wrap: true },
  {
    key: "vif",
    header: "VIF",
    mono: true,
    align: "right",
    cell: (r) => (typeof r.vif === "number" ? r.vif.toFixed(2) : "—"),
  },
  {
    key: "r2_explained_by_other_features",
    header: "R² explained",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.r2_explained_by_other_features === "number"
        ? r.r2_explained_by_other_features.toFixed(3)
        : "—",
  },
  {
    key: "severity",
    header: "Severity",
    cell: (r) => <SeverityChip value={safeStr(r.severity, "—")} />,
  },
];

interface LeakageRow extends Record<string, unknown> {
  feature?: string;
  column?: string;
  correlation?: number;
  mutual_information?: number;
  reason?: string;
}

const leakageColumns: DataTableColumn<LeakageRow>[] = [
  {
    key: "feature",
    header: "Feature",
    cell: (r) => safeStr(r.feature ?? r.column, "—"),
    wrap: true,
  },
  {
    key: "correlation",
    header: "Correlation",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.correlation === "number" ? r.correlation.toFixed(3) : "—",
  },
  {
    key: "mutual_information",
    header: "MI",
    mono: true,
    align: "right",
    cell: (r) =>
      typeof r.mutual_information === "number" ? r.mutual_information.toFixed(3) : "—",
  },
  {
    key: "reason",
    header: "Why suspect",
    cell: (r) => safeStr(r.reason, "—"),
    wrap: true,
  },
];

function SeverityChip({ value }: { value: string }) {
  const lower = value.toLowerCase();
  const cls =
    lower === "high"
      ? "border-rose-300/40 bg-rose-300/10 text-rose-200"
      : lower === "medium"
        ? "border-amber-300/40 bg-amber-300/10 text-amber-200"
        : lower === "low"
          ? "border-line bg-bg-2 text-ink-2"
          : "border-line bg-bg-2 text-ink-3";
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center rounded-full border px-2 font-mono text-[10px] uppercase tracking-[0.24em]",
        cls,
      )}
    >
      {value}
    </span>
  );
}

function kvFromModelDiagnostics(md: Record<string, unknown>): KeyValueItem[] {
  const items: KeyValueItem[] = [];
  const push = (label: string, value: KeyValueItem["value"], mono = true) => {
    if (value !== undefined && value !== null && value !== "") {
      items.push({ label, value, mono });
    }
  };
  push("Task type", safeStr(md.task_type, "—"));
  push("Target", safeStr(md.target_column, "—"));

  // Residual summary (regression).
  const res = safeObj(md.residual_summary, {} as Record<string, unknown>);
  for (const k of ["mean_residual", "median_residual", "std_residual", "min_residual", "max_residual", "mean_abs_residual"]) {
    if (typeof res[k] === "number") push(formatLabel(k), formatNumber(res[k] as number, 4));
  }

  // Holdout / classification metrics.
  for (const [k, v] of Object.entries(md)) {
    if (
      ["task_type", "target_column", "available", "message", "residual_summary", "warnings", "worst_residuals"].includes(k)
    ) continue;
    if (typeof v === "number") {
      push(formatLabel(k), formatNumber(v, 4));
    } else if (typeof v === "string") {
      push(formatLabel(k), v);
    }
  }
  return items;
}

function kvFromGenericObject(obj: Record<string, unknown>): KeyValueItem[] {
  return Object.entries(obj)
    .filter(([, v]) => v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0))
    .slice(0, 14)
    .map(([k, v]) => {
      let value: KeyValueItem["value"];
      if (typeof v === "number") value = formatNumber(v, 3);
      else if (typeof v === "string") value = v;
      else if (typeof v === "boolean") value = v ? "yes" : "no";
      else if (Array.isArray(v)) value = v.length ? `${v.length} entries` : "—";
      else if (v && typeof v === "object") value = `${Object.keys(v).length} entries`;
      else value = "—";
      return { label: formatLabel(k), value, mono: typeof v === "number" };
    });
}
