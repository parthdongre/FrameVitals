import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { safeArr, safeObj } from "@/lib/safe";
import type { TabComponentProps } from "./tabRegistry";

/**
 * Data Preview tab — renders the first N rows from `profile.preview`
 * (typically 15) as a sticky-header DataTable. Numeric columns get
 * tabular-nums for readability; cells are truncated unless the user wraps.
 */
export default function DataPreviewTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const profile = safeObj(t.profile, {} as Record<string, unknown>);
  const preview = safeArr<Record<string, unknown>>(profile.preview);
  const profileColumns = safeArr<string>(profile.columns);
  const numericCols = new Set(safeArr<string>(profile.numeric_columns));

  if (!preview.length) {
    return (
      <EmptyState
        title="No preview rows"
        hint="The backend did not include a preview slice for this dataset."
      />
    );
  }

  // Use the explicit profile.columns order when available, otherwise fall back
  // to the keys of the first row so we stay deterministic.
  const columnNames = profileColumns.length
    ? profileColumns
    : Object.keys(preview[0] ?? {});

  const columns: DataTableColumn<Record<string, unknown>>[] = columnNames.map((name) => ({
    key: name,
    header: name,
    mono: numericCols.has(name),
    align: numericCols.has(name) ? "right" : "left",
    cell: (row) => renderCell(row[name]),
  }));

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-6">
      <motion.section variants={staggerChild} className="flex flex-wrap items-baseline gap-3">
        <Eyebrow>Preview</Eyebrow>
        <span className="font-mono text-[11px] tabular-nums text-ink-3">
          {preview.length} row{preview.length === 1 ? "" : "s"} · {columnNames.length} column{columnNames.length === 1 ? "" : "s"}
        </span>
      </motion.section>

      <motion.div variants={staggerChild}>
        <DataTable
          columns={columns}
          rows={preview}
          rowKey={(_, i) => `preview-${i}`}
          maxHeight="70vh"
        />
      </motion.div>
    </motion.div>
  );
}

function renderCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return Math.abs(value) < 1
      ? value.toFixed(4)
      : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}
