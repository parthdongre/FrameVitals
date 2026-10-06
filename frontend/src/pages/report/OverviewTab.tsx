import { motion } from "framer-motion";
import { AnimatedNumber } from "@/components/site/AnimatedNumber";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { KeyValueGrid, type KeyValueItem } from "@/components/ui/KeyValueGrid";
import { HealthRadar } from "@/charts/HealthRadar";
import { DistributionHistogram } from "@/charts/DistributionHistogram";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, pickNum, pickStr, safeArr, safeNum, safeObj, safeStr } from "@/lib/safe";
import { formatCount, formatLabel, formatMs } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TabComponentProps } from "./tabRegistry";
import type { SignalItem } from "@/data/payload";

/**
 * Overview tab — top-line KPIs, signals, anomaly snapshot, distribution,
 * health radar, ML readiness, analysis selection, and a compact pipeline
 * timing strip. Every section is wrapped in a defensive read so a missing
 * payload field renders an `EmptyState` instead of throwing.
 */
export default function OverviewTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;

  const rows = safeNum(t.rows, pickNum(t.profile, "rows", 0));
  const cols = safeNum(t.columns, pickNum(t.profile, "columns", 0));
  const fileSize = safeStr(t.fileSize, "—");
  const lastScan = safeStr(t.lastScan, "—");
  const analysisDuration = safeNum(t.analysisDurationMs, 0);

  const metrics = safeArr<{ label?: string; value?: string; hint?: string }>(t.metrics);
  const dataTypes = safeArr<{ label?: string; count?: number }>(t.dataTypes);
  const signals = safeArr<SignalItem>(t.signals);

  const health = safeObj(t.health, {} as Record<string, unknown>);
  const ml = safeObj(t.mlReadiness, {} as Record<string, unknown>);

  const distribution = t.distribution;
  const rolesSummary = safeObj(t.rolesSummary, {} as Record<string, unknown>);
  const datasetSignals = safeObj(t.datasetSignals, {} as Record<string, unknown>);

  const anomalies = safeObj(t.anomaliesV2, {} as Record<string, unknown>);
  return (
    <motion.div
      variants={staggerParent}
      initial="initial"
      animate="animate"
      className="space-y-12"
    >
      {/* ------------------ Top-line KPI strip ----------------------- */}
      <motion.section variants={staggerChild} className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line sm:grid-cols-4">
        <Kpi label="Rows" value={<AnimatedNumber value={rows} />} sub={fileSize} />
        <Kpi label="Columns" value={<AnimatedNumber value={cols} />} sub={`${dataTypes.length} dtype${dataTypes.length === 1 ? "" : "s"}`} />
        <Kpi
          label="Health"
          value={<AnimatedNumber value={pickNum(health, "overall_score", 0)} suffix=" /100" />}
          sub={pickStr(health, "label", "")}
        />
        <Kpi
          label="ML readiness"
          value={<AnimatedNumber value={pickNum(ml, "score", 0)} suffix=" /100" />}
          sub={pickStr(ml, "label", "")}
        />
      </motion.section>

      {/* ------------------ Mission metrics --------------------------- */}
      {metrics.length ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Mission metrics</Eyebrow>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {metrics.map((m, i) => (
              <div
                key={`${i}-${m.label ?? "metric"}`}
                className="rounded-md border border-line bg-bg-1 p-4 transition-colors hover:border-line-strong"
              >
                <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">
                  {safeStr(m.label, "—")}
                </p>
                <p className="mt-2 text-[22px] font-semibold tabular-nums text-ink-1">
                  {safeStr(m.value, "—")}
                </p>
                {m.hint ? (
                  <p className="mt-1 text-[12px] leading-5 text-ink-3">{safeStr(m.hint, "")}</p>
                ) : null}
              </div>
            ))}
          </div>
        </motion.section>
      ) : null}

      {/* ------------------ Signals + Anomaly snapshot ---------------- */}
      <motion.section variants={staggerChild} className="grid gap-8 md:grid-cols-2">
        <div>
          <Eyebrow className="mb-3">Beacons</Eyebrow>
          {signals.length ? (
            <ul className="space-y-3">
              {signals.slice(0, 6).map((s, i) => (
                <li
                  key={`${s?.name ?? "signal"}-${i}`}
                  className="rounded-md border border-line bg-bg-1 p-4 transition-colors hover:border-line-strong"
                >
                  <div className="flex items-start gap-3 text-[13px]">
                    <SeverityChip severity={s?.severity} />
                    <div className="min-w-0">
                      <p className="font-medium text-ink-1">{safeStr(s?.name, "Signal")}</p>
                      <p className="mt-1 text-ink-2">{safeStr(s?.evidence, "—")}</p>
                      {s?.recommendation ? (
                        <p className="mt-1 text-accent">{safeStr(s.recommendation, "")}</p>
                      ) : null}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState compact title="No beacons raised" hint="Nothing currently needs your attention." />
          )}
        </div>

        <div>
          <Eyebrow className="mb-3">Irregular rows</Eyebrow>
          {pickNum(anomalies, "n_rows_scored", 0) > 0 ? (
            <div className="rounded-md border border-line bg-bg-1 p-5">
              <p className="text-[28px] font-semibold tabular-nums text-ink-1">
                {formatCount(pickNum(anomalies, "flagged_count", 0))}
                <span className="text-[16px] font-normal text-ink-3">
                  {" "}
                  / {formatCount(pickNum(anomalies, "n_rows_scored", 0))}
                </span>
              </p>
              <p className="mt-1 font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">rows worth reviewing</p>
              <p className="mt-3 text-[13px] text-ink-2">
                Prism found observations that differ meaningfully from the broader dataset.
              </p>
            </div>
          ) : (
            <EmptyState
              compact
              title="Irregularity view unavailable"
              hint="Run Prism at Standard depth or above to populate this view."
            />
          )}
        </div>
      </motion.section>

      {/* ------------------ Distribution + Health -------------------- */}
      <motion.section variants={staggerChild} className="grid gap-6 lg:grid-cols-2">
        <DistributionHistogram distribution={distribution} />
        <HealthRadar health={health} />
      </motion.section>

      {/* ------------------ ML readiness recommendations ------------- */}
      {safeArr(ml.recommendations).length > 0 ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">ML readiness recommendations</Eyebrow>
          <ul className="space-y-2 text-[14px] leading-7 text-ink-2">
            {safeArr<string>(ml.recommendations)
              .slice(0, 8)
              .map((rec, i) => (
                <li key={i} className="flex gap-3">
                  <span className="mt-2 h-px w-3 shrink-0 bg-accent" aria-hidden />
                  <span>{rec}</span>
                </li>
              ))}
          </ul>
        </motion.section>
      ) : null}

      {/* ------------------ Side rail: roles + dataset signals ------- */}
      {(isPresent(rolesSummary) || isPresent(datasetSignals)) ? (
        <motion.section variants={staggerChild} className="grid gap-6 md:grid-cols-2">
          {isPresent(rolesSummary) ? (
            <div>
              <Eyebrow className="mb-3">Roles summary</Eyebrow>
              <KeyValueGrid items={kvFromObject(rolesSummary)} cols={2} compact />
            </div>
          ) : null}
          {isPresent(datasetSignals) ? (
            <div>
              <Eyebrow className="mb-3">Dataset signals</Eyebrow>
              <KeyValueGrid items={kvFromObject(datasetSignals)} cols={2} compact />
            </div>
          ) : null}
        </motion.section>
      ) : null}

      {analysisDuration > 0 ? (
        <motion.section variants={staggerChild}>
          <Eyebrow className="mb-3">Prism completion</Eyebrow>
          <div className="rounded-md border border-line bg-bg-1 px-4 py-3">
            <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">
              Completed
            </p>
            <p className="mt-1 font-mono text-[13px] tabular-nums text-ink-1">
              {formatMs(analysisDuration)}
              {lastScan && lastScan !== "—" ? <span className="text-ink-3"> · {lastScan}</span> : null}
            </p>
          </div>
        </motion.section>
      ) : null}

  );
}

/* --- helpers -------------------------------------------------------- */

function Kpi({
  label,
  value,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
}) {
  return (
    <div className="bg-bg-1 p-5">
      <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">{label}</p>
      <p className="mt-2 text-[28px] font-semibold tabular-nums text-ink-1">{value}</p>
      {sub ? (
        <p className="mt-1 text-[11px] uppercase tracking-[0.24em] text-ink-3">{sub}</p>
      ) : null}
    </div>
  );
}

function SeverityChip({ severity }: { severity: unknown }) {
  const raw = safeStr(severity, "info").toLowerCase();
  const cls =
    raw === "critical"
      ? "border-rose-300/40 bg-rose-300/10 text-rose-200"
      : raw === "high"
        ? "border-orange-300/40 bg-orange-300/10 text-orange-200"
        : raw === "medium"
          ? "border-amber-300/40 bg-amber-300/10 text-amber-200"
          : raw === "low"
            ? "border-line bg-bg-2 text-ink-2"
            : "border-line bg-bg-2 text-ink-2";

  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center rounded-full border px-2 font-mono text-[10px] uppercase tracking-[0.24em]",
        cls,
      )}
    >
      {raw}
    </span>
  );
}

function kvFromObject(obj: Record<string, unknown>): KeyValueItem[] {
  return Object.entries(obj)
    .filter(([, v]) => v !== null && v !== undefined && !(Array.isArray(v) && v.length === 0))
    .slice(0, 12)
    .map(([k, v]) => ({
      label: formatLabel(k),
      value: renderValue(v),
      mono: typeof v === "number",
    }));
}

function renderValue(v: unknown): React.ReactNode {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isFinite(v) ? v.toLocaleString() : "—";
  if (typeof v === "string") return v;
  if (Array.isArray(v)) {
    if (v.length === 0) return "—";
    return v
      .slice(0, 6)
      .map((x) => (typeof x === "string" ? x : JSON.stringify(x)))
      .join(", ") + (v.length > 6 ? "…" : "");
  }
  if (typeof v === "object") {
    const keys = Object.keys(v as object);
    return keys.length ? `${keys.length} entries` : "—";
  }
  return String(v);
}
