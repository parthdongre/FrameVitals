import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { DataTable, type DataTableColumn } from "@/components/ui/DataTable";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeArr, safeNum, safeObj, safeStr } from "@/lib/safe";
import { formatCount, formatNumber } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TabComponentProps } from "./tabRegistry";

interface CleaningAction extends Record<string, unknown> {
  action?: string;
  details?: string;
}

function getBackendBaseUrl(): string {
  const env = (import.meta as any).env ?? {};
  const configured = (env.VITE_BACKEND_URL ?? "").toString().trim();
  if (configured) return configured.replace(/\/$/, "");
  if (typeof window !== "undefined" && window.location.port === "5173") {
    return "http://127.0.0.1:5055";
  }
  return "";
}

/**
 * Cleaning tab — before/after KPIs, the action log, and the cleaned-CSV
 * download link from `downloadLinks.cleaned`.
 */
export default function CleaningTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const cleaning = safeObj(t.cleaning, {} as Record<string, unknown>);
  const downloadLinks = safeObj(t.downloadLinks, {} as Record<string, unknown>);
  const cleanedHref = safeStr(downloadLinks.cleaned, "");

  if (!isPresent(cleaning)) {
    return (
      <EmptyState
        title="Cleaning summary not available for this dataset"
        hint="The cleaner only runs in standard or deeper modes. Re-run with one of those modes to populate this tab."
      />
    );
  }

  const before = safeObj(cleaning.before_health, {} as Record<string, unknown>);
  const after = safeObj(cleaning.after_health, {} as Record<string, unknown>);
  const missingBefore = safeNum(cleaning.missing_before, 0);
  const missingAfter = safeNum(cleaning.missing_after, 0);
  const dupBefore = safeNum(cleaning.duplicates_before, 0);
  const dupAfter = safeNum(cleaning.duplicates_after, 0);
  const actions = safeArr<CleaningAction>(cleaning.actions);

  const healthBefore = safeNum(before.overall_score, 0);
  const healthAfter = safeNum(after.overall_score, 0);
  const healthDelta = healthAfter - healthBefore;

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-10">
      <motion.section
        variants={staggerChild}
        className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-4"
      >
        <DeltaCard
          label="Missing values"
          before={missingBefore}
          after={missingAfter}
          formatter={formatCount}
        />
        <DeltaCard
          label="Duplicate rows"
          before={dupBefore}
          after={dupAfter}
          formatter={formatCount}
        />
        <DeltaCard
          label="Health score"
          before={healthBefore}
          after={healthAfter}
          formatter={(v) => formatNumber(v, 1)}
          higherIsBetter
        />
        <div className="bg-bg-1 p-5">
          <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">Cleaned CSV</p>
          {cleanedHref ? (
            <a
              href={`${getBackendBaseUrl()}${cleanedHref}`}
              className="btn-ghost mt-3"
              download
            >
              Download cleaned CSV
            </a>
          ) : (
            <p className="mt-3 text-[12px] text-ink-3">Path unavailable.</p>
          )}
          <p className="mt-2 font-mono text-[11px] tabular-nums text-ink-3">
            {actions.length} action{actions.length === 1 ? "" : "s"} applied
          </p>
        </div>
      </motion.section>

      <motion.section variants={staggerChild}>
        <Eyebrow className="mb-3">
          {healthDelta >= 0 ? "Improvement" : "Regression"} ·{" "}
          <span className="font-mono text-accent">
            {healthDelta >= 0 ? "+" : ""}
            {formatNumber(healthDelta, 1)}
          </span>{" "}
          health points
        </Eyebrow>
        <p className="max-w-3xl text-[14px] leading-7 text-ink-2">
          The cleaner imputes missing values (median for numeric, mode for categorical), removes
          exact duplicate rows, clips extreme outliers, and rescores health on the cleaned dataset.
          The diff shows the gain.
        </p>
      </motion.section>

      <motion.section variants={staggerChild}>
        <Eyebrow className="mb-3">Action log</Eyebrow>
        {actions.length ? (
          <DataTable
            columns={actionColumns}
            rows={actions}
            rowKey={(_, i) => `action-${i}`}
            maxHeight="60vh"
          />
        ) : (
          <EmptyState
            compact
            title="No cleaning actions were needed"
            hint="The dataset arrived clean — no missing values, duplicates, or outliers to fix."
          />
        )}
      </motion.section>
    </motion.div>
  );
}

const actionColumns: DataTableColumn<CleaningAction>[] = [
  {
    key: "action",
    header: "Action",
    cell: (r) => safeStr(r.action, "—"),
    wrap: true,
  },
  {
    key: "details",
    header: "Details",
    cell: (r) => safeStr(r.details, "—"),
    wrap: true,
  },
];

function DeltaCard({
  label,
  before,
  after,
  formatter,
  higherIsBetter = false,
}: {
  label: string;
  before: number;
  after: number;
  formatter: (v: number) => string;
  higherIsBetter?: boolean;
}) {
  const delta = after - before;
  const improved = higherIsBetter ? delta >= 0 : delta <= 0;
  const showDelta = before !== after;
  return (
    <div className="bg-bg-1 p-5">
      <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">{label}</p>
      <div className="mt-2 flex items-baseline gap-2 text-[20px] tabular-nums">
        <span className="text-ink-3 line-through decoration-ink-4">{formatter(before)}</span>
        <span className="text-ink-3">→</span>
        <span className="font-semibold text-ink-1">{formatter(after)}</span>
      </div>
      {showDelta ? (
        <p
          className={cn(
            "mt-1 font-mono text-[11px] tabular-nums",
            improved ? "text-accent" : "text-rose-300",
          )}
        >
          {delta > 0 ? "+" : ""}
          {formatter(delta)}
        </p>
      ) : (
        <p className="mt-1 font-mono text-[11px] text-ink-3">no change</p>
      )}
    </div>
  );
}
