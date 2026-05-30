import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { KeyValueGrid, type KeyValueItem } from "@/components/ui/KeyValueGrid";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { MissingnessBars } from "@/charts/MissingnessBars";
import { CorrelationHeatmap } from "@/charts/CorrelationHeatmap";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeArr, safeNum, safeObj, safeStr } from "@/lib/safe";
import { formatLabel, formatPercent } from "@/lib/format";
import type { TabComponentProps } from "./tabRegistry";

/**
 * Profile tab — schema, dtypes, missingness, duplicates, correlations,
 * column-utility, and the per-column drawer with role chips.
 */
export default function ProfileTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const profile = safeObj(t.profile, {} as Record<string, unknown>);
  const advanced = safeObj(t.advanced, {} as Record<string, unknown>);
  const columnRoles = safeObj(t.columnRoles, {} as Record<string, unknown>);

  const dtypes = safeObj(profile.dtypes, {} as Record<string, unknown>);
  const numericCols = safeArr<string>(profile.numeric_columns);
  const categoricalCols = safeArr<string>(profile.categorical_columns);
  const dateCols = safeArr<string>(profile.date_columns);
  const totalRows = safeNum(profile.shape && (profile.shape as { rows?: number }).rows, safeNum(t.rows, 0));
  const duplicateRows = safeNum(profile.duplicate_rows, 0);

  const missingPercent = safeObj(profile.missing_percent, {} as Record<string, unknown>);
  const missingCounts = safeObj(profile.missing_counts, {} as Record<string, unknown>);
  const correlations = profile.correlations;

  const utilityRows = useColumnUtility(advanced.column_utility, columnRoles, dtypes);
  const dtypesRows = useDtypeRows(dtypes, numericCols, categoricalCols, dateCols, missingPercent, columnRoles);

  const utilityCols: DataTableColumn<UtilityRow>[] = [
    { key: "column", header: "Column", wrap: true },
    { key: "utility", header: "Utility", mono: true, align: "right" },
    { key: "rationale", header: "Rationale", wrap: true },
    { key: "roles", header: "Roles" },
  ];

  const dtypeCols: DataTableColumn<DtypeRow>[] = [
    { key: "column", header: "Column", wrap: true },
    { key: "dtype", header: "Dtype", mono: true },
    { key: "kind", header: "Kind" },
    { key: "missing", header: "% missing", mono: true, align: "right" },
    { key: "roles", header: "Roles", wrap: true },
  ];

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-12">
      <motion.section variants={staggerChild}>
        <Eyebrow className="mb-3">Schema</Eyebrow>
        <KeyValueGrid
          cols={4}
          items={[
            { label: "Rows", value: totalRows.toLocaleString(), mono: true },
            { label: "Columns", value: Object.keys(dtypes).length || safeNum(t.columns, 0), mono: true },
            { label: "Numeric", value: numericCols.length, mono: true },
            { label: "Categorical", value: categoricalCols.length, mono: true },
            { label: "Date", value: dateCols.length, mono: true },
            { label: "Duplicates", value: duplicateRows.toLocaleString(), mono: true },
            { label: "Missing rate", value: overallMissingRate(missingPercent), mono: true },
            {
              label: "Freshness",
              value: safeStr(safeObj(advanced.freshness, {} as Record<string, unknown>).label, "—"),
            },
          ]}
        />
      </motion.section>

      <motion.section variants={staggerChild} className="grid gap-6 lg:grid-cols-2">
        <MissingnessBars percent={missingPercent} total={totalRows} />
        {!isPresent(missingPercent) && Object.keys(missingCounts).length ? (
          <MissingnessBars percent={missingCounts} total={totalRows} />
        ) : (
          <CorrelationHeatmap correlations={correlations} />
        )}
      </motion.section>

      {isPresent(missingPercent) && Object.keys(safeObj(correlations, {} as Record<string, unknown>)).length ? (
        <motion.section variants={staggerChild}>
          <CorrelationHeatmap correlations={correlations} />
        </motion.section>
      ) : null}

      <motion.section variants={staggerChild}>
        <Eyebrow className="mb-3">Per-column schema</Eyebrow>
        {dtypesRows.length ? (
          <DataTable
            columns={dtypeCols}
            rows={dtypesRows}
            rowKey={(r) => r.column}
            maxHeight="60vh"
          />
        ) : (
          <EmptyState compact title="Schema unavailable" />
        )}
      </motion.section>

      {utilityRows.length ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Column utility</Eyebrow>
          <p className="mb-3 text-[13px] leading-6 text-ink-3">
            Estimated downstream usefulness per column, computed from cardinality, missingness,
            and role tags.
          </p>
          <DataTable columns={utilityCols} rows={utilityRows} rowKey={(r) => r.column} maxHeight="60vh" />
        </motion.section>
      ) : null}

      {isPresent(advanced.freshness) ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Advanced freshness signals</Eyebrow>
          <KeyValueGrid items={kvFromObject(safeObj(advanced.freshness, {}))} cols={2} compact />
        </motion.section>
      ) : null}
    </motion.div>
  );
}

/* --- helpers ---------------------------------------------------------- */

interface DtypeRow extends Record<string, unknown> {
  column: string;
  dtype: string;
  kind: string;
  missing: string;
  roles: string;
}

function useDtypeRows(
  dtypes: Record<string, unknown>,
  numericCols: string[],
  categoricalCols: string[],
  dateCols: string[],
  missingPercent: Record<string, unknown>,
  roles: Record<string, unknown>,
): DtypeRow[] {
  const numericSet = new Set(numericCols);
  const categoricalSet = new Set(categoricalCols);
  const dateSet = new Set(dateCols);
  return Object.entries(dtypes).map(([col, dtype]) => {
    const kind =
      dateSet.has(col)
        ? "date"
        : numericSet.has(col)
          ? "numeric"
          : categoricalSet.has(col)
            ? "categorical"
            : "other";
    const pct = safeNum(missingPercent[col], 0);
    const roleEntry = safeObj(roles[col], {} as Record<string, unknown>);
    const tags = safeArr<string>(roleEntry.roles).slice(0, 4).join(", ");
    return {
      column: col,
      dtype: String(dtype ?? "—"),
      kind,
      missing: pct > 0 ? formatPercent(pct / 100, 2) : "0%",
      roles: tags || "—",
    };
  });
}

interface UtilityRow extends Record<string, unknown> {
  column: string;
  utility: string;
  rationale: string;
  roles: string;
}

function useColumnUtility(
  utility: unknown,
  roles: Record<string, unknown>,
  dtypes: Record<string, unknown>,
): UtilityRow[] {
  const map = safeObj(utility, {} as Record<string, unknown>);
  const cols = Object.keys(map).length ? Object.keys(map) : Object.keys(dtypes);
  if (!cols.length) return [];
  return cols
    .map((col) => {
      const u = safeObj(map[col], {} as Record<string, unknown>);
      const score = safeNum(u.score ?? u.utility, NaN);
      const rationale = safeStr(u.rationale ?? u.reason, "");
      const roleEntry = safeObj(roles[col], {} as Record<string, unknown>);
      const tags = safeArr<string>(roleEntry.roles).slice(0, 4).join(", ");
      return {
        column: col,
        utility: Number.isFinite(score) ? score.toFixed(2) : "—",
        rationale,
        roles: tags || "—",
        _sort: Number.isFinite(score) ? score : -1,
      } as UtilityRow & { _sort: number };
    })
    .sort((a, b) => b._sort - a._sort)
    .map(({ _sort: _, ...row }) => row);
}

function overallMissingRate(missingPercent: Record<string, unknown>): string {
  const values = Object.values(missingPercent)
    .map((v) => safeNum(v, NaN))
    .filter((v) => Number.isFinite(v));
  if (!values.length) return "0%";
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return formatPercent(mean / 100, 2);
}

function kvFromObject(obj: Record<string, unknown>): KeyValueItem[] {
  return Object.entries(obj)
    .filter(([, v]) => v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0))
    .slice(0, 12)
    .map(([k, v]) => ({
      label: formatLabel(k),
      value: typeof v === "object" ? JSON.stringify(v) : String(v),
      mono: typeof v === "number",
    }));
}
