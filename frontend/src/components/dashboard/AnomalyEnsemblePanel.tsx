import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface TopRow {
  row_index: number | string;
  ensemble: number;
  [detector: string]: number | string;
}

interface AnomalyEnsemblePanelProps {
  anomalies?: {
    available?: boolean;
    message?: string;
    n_rows_scored?: number;
    used_columns?: string[];
    detectors_run?: string[];
    threshold?: number;
    contamination?: number;
    flagged_count?: number;
    ensemble_summary?: {
      min: number;
      mean: number;
      median: number;
      max: number;
      p95: number;
      p99: number;
    };
    top_rows?: TopRow[];
    interpretation?: string;
  };
}

function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    return v.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return String(v);
}

function severityForScore(s: number): "high" | "medium" | "low" | "muted" {
  if (s >= 0.85) return "high";
  if (s >= 0.6) return "medium";
  if (s >= 0.3) return "low";
  return "muted";
}

function colorForScore(s: number): string {
  // Continuous gradient from slate (low) → amber → rose (high)
  if (s >= 0.85) return "rgba(244,63,94,0.85)"; // rose
  if (s >= 0.6) return "rgba(251,146,60,0.85)"; // orange
  if (s >= 0.4) return "rgba(245,158,11,0.7)"; // amber
  if (s >= 0.2) return "rgba(56,189,248,0.55)"; // sky
  return "rgba(148,163,184,0.35)"; // slate
}

export function AnomalyEnsemblePanel({ anomalies }: AnomalyEnsemblePanelProps) {
  if (!anomalies) return null;

  if (!anomalies.available) {
    return (
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="text-lg text-slate-50">
            <span className="text-cyan-300">Irregular rows</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            Prism did not produce an irregularity view for this dataset.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-6 text-sm text-slate-400">
            {anomalies.message ?? "Anomaly ensemble unavailable for this dataset."}
          </div>
        </CardContent>
      </Card>
    );
  }

  const topRows = anomalies.top_rows ?? [];
  const summary = anomalies.ensemble_summary;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="text-lg text-slate-50">
            <span className="text-cyan-300">Irregular rows</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            Rows that differ meaningfully from the broader dataset and deserve review.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5 p-6">
          <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
            <SummaryStat
              label="Rows scored"
              value={fmt(anomalies.n_rows_scored)}
              tone="slate"
            />
            <SummaryStat label="Flagged" value={fmt(anomalies.flagged_count)} tone="violet" />
            <SummaryStat label="Mean" value={fmt(summary?.mean)} tone="cyan" />
            <SummaryStat label="Median" value={fmt(summary?.median)} tone="cyan" />
            <SummaryStat label="P95" value={fmt(summary?.p95)} tone="cyan" />
            <SummaryStat label="P99" value={fmt(summary?.p99)} tone="cyan" />
          </div>


          {topRows.length > 0 ? (
            <div className="overflow-x-auto rounded-2xl border border-white/5 bg-white/[0.02]">
              <table className="w-full border-separate border-spacing-0 text-left text-sm">
                <thead className="bg-white/[0.02] text-[11px] uppercase tracking-[0.3em] text-slate-500">
                  <tr>
                    <th className="border-b border-white/5 px-3 py-2 font-semibold">#</th>
                    <th className="border-b border-white/5 px-3 py-2 font-semibold">Row</th>
                    <th className="border-b border-white/5 px-3 py-2 font-semibold">Severity</th>
                    <th className="border-b border-white/5 px-3 py-2 text-right font-semibold text-cyan-200">
                      score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {topRows.slice(0, 12).map((row, idx) => {
                    const sev = severityForScore(row.ensemble);
                    const sevBadge =
                      sev === "high" ? "high" : sev === "medium" ? "medium" : sev === "low" ? "low" : "muted";
                    return (
                      <tr
                        key={`${row.row_index}-${idx}`}
                        className="border-b border-white/5 odd:bg-white/[0.01] even:bg-transparent"
                      >
                        <td className="border-b border-white/5 px-3 py-2 font-mono text-slate-400">
                          {idx + 1}
                        </td>
                        <td className="border-b border-white/5 px-3 py-2 font-mono text-slate-200">
                          {row.row_index}
                        </td>
                        <td className="border-b border-white/5 px-3 py-2">
                          <Badge
                            variant={
                              sevBadge === "high"
                                ? "violet"
                                : sevBadge === "medium"
                                ? "cyan"
                                : "muted"
                            }
                          >
                            {sevBadge}
                          </Badge>
                        </td>
                        <td className="border-b border-white/5 px-3 py-2 text-right font-mono font-semibold text-cyan-200">
                          {fmt(row.ensemble, 3)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No irregular rows were ranked.</p>
          )}


        </CardContent>
      </Card>
    </motion.div>
  );
}

function SummaryStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "cyan" | "violet" | "slate";
}) {
  const toneClasses = {
    cyan: "border-cyan-400/15 bg-cyan-500/5 text-cyan-100",
    violet: "border-violet-400/15 bg-violet-500/5 text-violet-100",
    slate: "border-white/10 bg-white/[0.03] text-slate-100",
  } as const;
  return (
    <div className={`rounded-2xl border p-3 ${toneClasses[tone]}`}>
      <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-lg text-slate-50">{value}</p>
    </div>
  );
}
