import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface GlobalImportance {
  feature: string;
  importance: number;
}

interface PerRowStory {
  rank: number;
  row_index: number | string;
  top_contributions: Array<{ feature: string; shap_value: number }>;
}

interface ExplainabilityPanelProps {
  explainability?: {
    available?: boolean;
    message?: string;
    method?: string;
    model?: string;
    task_type?: string;
    global_importance?: GlobalImportance[];
    per_row_stories?: PerRowStory[];
    summary_chart_path?: string | null;
    n_test_rows_explained?: number;
    permutation_importance?: {
      available?: boolean;
      method?: string;
      top_features?: Array<{ feature: string; importance: number; std?: number }>;
    };
    errors?: string[];
  };
  backendBaseUrl?: string;
}

function fmt(value: unknown, digits = 4): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return String(value);
}

function fmtSigned(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const formatted = Math.abs(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
  return `${value >= 0 ? "+" : "−"}${formatted}`;
}

function buildBar(value: number, max: number, signed = false) {
  if (max === 0) return { width: 0, side: "right" as const };
  const pct = Math.min(100, (Math.abs(value) / max) * 100);
  const side: "left" | "right" = signed ? (value < 0 ? "left" : "right") : "right";
  return { width: pct, side };
}

export function ExplainabilityPanel({ explainability, backendBaseUrl = "" }: ExplainabilityPanelProps) {
  if (!explainability) return null;

  if (!explainability.available) {
    return (
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Model explainability</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">SHAP</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            SHAP-based global + per-row attribution for the leaderboard winner.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-6 text-sm text-slate-400">
            {explainability.message ?? "Run analysis with a target column to unlock SHAP explanations."}
          </div>
        </CardContent>
      </Card>
    );
  }

  const global = Array.isArray(explainability.global_importance)
    ? explainability.global_importance.slice(0, 10)
    : [];
  const stories = Array.isArray(explainability.per_row_stories) ? explainability.per_row_stories : [];
  const maxGlobal = Math.max(...global.map((g) => Math.abs(g.importance || 0)), 0);
  const chartPath = explainability.summary_chart_path
    ? `${backendBaseUrl}/${String(explainability.summary_chart_path).replace(/^\/?(static\/)?/, "static/")}`
    : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Model explainability</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">SHAP</span>
          </CardTitle>
          <CardDescription className="flex flex-wrap items-center gap-2 text-slate-400">
            <span>method:</span>
            <Badge variant="outline" className="border-cyan-400/20 text-cyan-100">
              {explainability.method}
            </Badge>
            <span>·</span>
            <span>model:</span>
            <Badge variant="outline" className="border-violet-400/20 text-violet-100">
              {explainability.model}
            </Badge>
            <span>·</span>
            <span>{explainability.n_test_rows_explained ?? 0} test rows explained</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6 p-6">
          <div className="grid gap-6 xl:grid-cols-2">
            <section>
              <p className="mb-3 text-xs uppercase tracking-[0.28em] text-slate-500">
                Global importance (top 10)
              </p>
              <div className="space-y-2">
                {global.length > 0 ? (
                  global.map((row) => {
                    const bar = buildBar(row.importance ?? 0, maxGlobal);
                    return (
                      <div
                        key={row.feature}
                        className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-sm text-slate-100 truncate">{row.feature}</span>
                          <span className="font-mono text-sm text-cyan-200">
                            {fmt(row.importance)}
                          </span>
                        </div>
                        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/[0.04]">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-cyan-300"
                            style={{ width: `${bar.width}%` }}
                          />
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p className="text-sm text-slate-500">No global importance scores available.</p>
                )}
              </div>
            </section>

            <section>
              <p className="mb-3 text-xs uppercase tracking-[0.28em] text-slate-500">
                Top SHAP summary plot
              </p>
              {chartPath ? (
                <div className="overflow-hidden rounded-2xl border border-white/5 bg-space-950/80">
                  <img
                    src={chartPath}
                    alt="SHAP summary"
                    className="h-auto w-full object-contain"
                  />
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-6 text-sm text-slate-400">
                  No SHAP summary plot was produced.
                </div>
              )}
            </section>
          </div>

          {stories.length > 0 ? (
            <section>
              <p className="mb-3 text-xs uppercase tracking-[0.28em] text-slate-500">
                Per-row explanations · top 3 most-explained rows
              </p>
              <div className="grid gap-3 xl:grid-cols-3">
                {stories.map((story) => {
                  const localMax = Math.max(
                    ...story.top_contributions.map((c) => Math.abs(c.shap_value || 0)),
                    0,
                  );
                  return (
                    <div
                      key={`${story.rank}-${story.row_index}`}
                      className="rounded-2xl border border-white/5 bg-white/[0.02] p-4"
                    >
                      <div className="flex items-center justify-between text-xs text-slate-500">
                        <span>Rank #{story.rank}</span>
                        <span className="font-mono">row {story.row_index}</span>
                      </div>
                      <div className="mt-3 space-y-2">
                        {story.top_contributions.map((c) => {
                          const bar = buildBar(c.shap_value ?? 0, localMax, true);
                          const positive = (c.shap_value ?? 0) >= 0;
                          return (
                            <div key={c.feature}>
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-sm text-slate-200 truncate">{c.feature}</span>
                                <span
                                  className={`font-mono text-xs ${
                                    positive ? "text-emerald-300" : "text-rose-300"
                                  }`}
                                >
                                  {fmtSigned(c.shap_value ?? 0)}
                                </span>
                              </div>
                              <div className="relative mt-1 flex h-1.5 w-full justify-center overflow-hidden rounded-full bg-white/[0.04]">
                                <div className="relative h-full w-1/2">
                                  {bar.side === "left" ? (
                                    <div
                                      className="absolute right-0 top-0 h-full rounded-full bg-rose-400/70"
                                      style={{ width: `${bar.width}%` }}
                                    />
                                  ) : null}
                                </div>
                                <div className="relative h-full w-1/2">
                                  {bar.side === "right" ? (
                                    <div
                                      className="absolute left-0 top-0 h-full rounded-full bg-emerald-400/70"
                                      style={{ width: `${bar.width}%` }}
                                    />
                                  ) : null}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          ) : null}

          {explainability.errors && explainability.errors.length > 0 ? (
            <p className="text-xs text-amber-300">
              Notes: {explainability.errors.join(" · ")}
            </p>
          ) : null}
        </CardContent>
      </Card>
    </motion.div>
  );
}
