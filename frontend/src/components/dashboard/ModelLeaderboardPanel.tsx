import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface LeaderboardRow {
  model: string;
  primary_score?: number | null;
  fit_time_s?: number | null;
  cv_total_s?: number | null;
  n_splits?: number;
  error?: string;
  [key: string]: unknown;
}

interface ModelLeaderboardPanelProps {
  leaderboard?: {
    available?: boolean;
    message?: string;
    task_type?: string;
    target_column?: string;
    primary_metric?: string;
    n_rows?: number;
    n_features?: number;
    leaderboard?: LeaderboardRow[];
    winner?: {
      model?: string;
      primary_score?: number | null;
      fit_time_s?: number | null;
      holdout?: Record<string, unknown> & { available?: boolean };
    } | null;
    dropped_columns?: Array<{ column: string; reason: string }>;
  };
}

function fmt(value: unknown, digits = 4): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return String(value);
}

function severityForScore(score: number | null | undefined, taskType?: string): "high" | "medium" | "low" | "muted" {
  if (score === null || score === undefined || !Number.isFinite(score)) return "muted";
  if (taskType === "classification") {
    if (score >= 0.85) return "high";
    if (score >= 0.7) return "medium";
    return "low";
  }
  if (score >= 0.7) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

export function ModelLeaderboardPanel({ leaderboard }: ModelLeaderboardPanelProps) {
  if (!leaderboard) {
    return null;
  }

  if (!leaderboard.available) {
    return (
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Model leaderboard</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">ML</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            Cross-validated model leaderboard with calibrated holdout metrics.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-6 text-sm text-slate-400">
            {leaderboard.message ?? "Select a target column to unlock the model leaderboard."}
          </div>
        </CardContent>
      </Card>
    );
  }

  const rows = Array.isArray(leaderboard.leaderboard) ? leaderboard.leaderboard : [];
  const winner = leaderboard.winner ?? null;
  const taskType = leaderboard.task_type ?? "?";
  const primaryMetric = leaderboard.primary_metric ?? "score";
  const holdout = winner?.holdout ?? {};

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Model leaderboard</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">ML · {taskType.toUpperCase()}</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            5-fold cross-validated comparison · target{" "}
            <span className="font-mono text-slate-200">{leaderboard.target_column}</span> · primary metric{" "}
            <span className="font-mono text-cyan-200">{primaryMetric}</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5 p-6">
          {winner ? (
            <div className="rounded-2xl border border-cyan-400/20 bg-cyan-500/[0.04] p-4">
              <div className="flex flex-wrap items-center gap-3">
                <Badge variant="outline" className="border-cyan-400/30 text-cyan-100">
                  Winner
                </Badge>
                <span className="text-base font-semibold text-slate-50">{winner.model}</span>
                <span className="font-mono text-cyan-200">
                  {primaryMetric}: {fmt(winner.primary_score)}
                </span>
                {winner.fit_time_s !== undefined ? (
                  <span className="text-xs text-slate-400">fit {fmt(winner.fit_time_s, 2)}s · CV avg</span>
                ) : null}
              </div>

              {holdout?.available ? (
                <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                  {Object.entries(holdout)
                    .filter(([k, v]) => k !== "available" && k !== "residual_summary" && typeof v !== "object")
                    .slice(0, 8)
                    .map(([k, v]) => (
                      <div
                        key={k}
                        className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2"
                      >
                        <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">
                          {k.replace(/_/g, " ")}
                        </p>
                        <p className="font-mono text-sm text-slate-100">{fmt(v)}</p>
                      </div>
                    ))}
                </div>
              ) : null}
            </div>
          ) : null}

          <div className="overflow-x-auto rounded-2xl border border-white/5 bg-white/[0.02]">
            <table className="w-full border-separate border-spacing-0 text-left text-sm">
              <thead className="bg-white/[0.02] text-[11px] uppercase tracking-[0.3em] text-slate-500">
                <tr>
                  <th className="border-b border-white/5 px-4 py-3 font-semibold">#</th>
                  <th className="border-b border-white/5 px-4 py-3 font-semibold">Model</th>
                  <th className="border-b border-white/5 px-4 py-3 font-semibold">CV {primaryMetric}</th>
                  <th className="border-b border-white/5 px-4 py-3 font-semibold">Std</th>
                  <th className="border-b border-white/5 px-4 py-3 font-semibold">Fit time</th>
                  <th className="border-b border-white/5 px-4 py-3 font-semibold">Splits</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => {
                  const stdKey = primaryMetric.startsWith("neg_")
                    ? `${primaryMetric.slice(4)}_std`
                    : `${primaryMetric}_std`;
                  const stdVal = row[stdKey] as number | undefined;
                  const sev = severityForScore(row.primary_score ?? null, taskType);
                  const sevClass =
                    sev === "high"
                      ? "text-emerald-300"
                      : sev === "medium"
                      ? "text-amber-300"
                      : sev === "low"
                      ? "text-rose-300"
                      : "text-slate-400";

                  return (
                    <tr
                      key={`${row.model}-${index}`}
                      className="border-b border-white/5 odd:bg-white/[0.01] even:bg-transparent"
                    >
                      <td className="border-b border-white/5 px-4 py-3 font-mono text-slate-400">
                        {index + 1}
                      </td>
                      <td className="border-b border-white/5 px-4 py-3 text-slate-100">
                        <div className="flex flex-wrap items-center gap-2">
                          <span>{row.model}</span>
                          {winner?.model === row.model ? (
                            <Badge variant="outline" className="border-cyan-400/30 text-cyan-100">
                              winner
                            </Badge>
                          ) : null}
                          {row.error ? (
                            <Badge variant="violet">error</Badge>
                          ) : null}
                        </div>
                      </td>
                      <td className={`border-b border-white/5 px-4 py-3 font-mono ${sevClass}`}>
                        {row.error ? "—" : fmt(row.primary_score)}
                      </td>
                      <td className="border-b border-white/5 px-4 py-3 font-mono text-slate-400">
                        {fmt(stdVal)}
                      </td>
                      <td className="border-b border-white/5 px-4 py-3 font-mono text-slate-400">
                        {row.fit_time_s !== undefined ? `${fmt(row.fit_time_s, 2)}s` : "—"}
                      </td>
                      <td className="border-b border-white/5 px-4 py-3 font-mono text-slate-400">
                        {row.n_splits ?? "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex flex-wrap gap-2 text-xs text-slate-500">
            <span>n_rows: {leaderboard.n_rows ?? "?"}</span>
            <span>·</span>
            <span>n_features: {leaderboard.n_features ?? "?"}</span>
            {leaderboard.dropped_columns && leaderboard.dropped_columns.length > 0 ? (
              <>
                <span>·</span>
                <span>dropped: {leaderboard.dropped_columns.length}</span>
              </>
            ) : null}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
