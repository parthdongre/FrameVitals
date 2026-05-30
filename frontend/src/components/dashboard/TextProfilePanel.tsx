import { useState } from "react";
import { motion } from "framer-motion";
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface NGramRow {
  term: string;
  count: number;
}

interface ColumnProfile {
  available?: boolean;
  reason?: string;
  n_rows?: number;
  length_stats?: { avg?: number; min?: number; max?: number; median?: number };
  tokens?: { total_tokens?: number; vocab_size?: number; lexical_diversity?: number };
  avg_sentence_length?: number;
  language?: { language?: string; confidence?: number };
  top_unigrams?: NGramRow[];
  top_bigrams?: NGramRow[];
  patterns?: {
    counts?: Record<string, number>;
    examples?: Record<string, string[]>;
    total_rows?: number;
  };
  sentiment_lite?: {
    positive_ratio?: number;
    negative_ratio?: number;
    polarity?: number;
    n_tokens?: number;
  };
  lsa?: {
    available?: boolean;
    reason?: string;
    method?: string;
    explained_variance_ratio?: number[];
    points?: Array<{ x: number | null; y: number | null; preview: string }>;
  };
}

interface TextProfilePanelProps {
  textProfile?: {
    available?: boolean;
    reason?: string;
    error?: string;
    detected_columns?: string[];
    profiled_columns?: string[];
    stopwords_source?: string;
    profiles?: Record<string, ColumnProfile>;
  };
}

function fmt(value: unknown, digits = 4): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return String(value);
}

const HC_DARK_THEME: Highcharts.Options = {
  chart: { backgroundColor: "transparent", style: { fontFamily: "Inter, system-ui, sans-serif" } },
  credits: { enabled: false },
  legend: { itemStyle: { color: "#cbd5e1" } },
  xAxis: {
    labels: { style: { color: "#94a3b8", fontSize: "11px" } },
    lineColor: "rgba(255,255,255,0.05)",
    tickColor: "rgba(255,255,255,0.05)",
  },
  yAxis: {
    labels: { style: { color: "#94a3b8", fontSize: "11px" } },
    gridLineColor: "rgba(255,255,255,0.04)",
    title: { text: undefined },
  },
};

function buildLsaScatter(lsa?: ColumnProfile["lsa"]): Highcharts.Options | null {
  if (!lsa?.available || !lsa.points || lsa.points.length === 0) return null;

  const data = lsa.points
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .map((p) => ({
      x: p.x as number,
      y: p.y as number,
      preview: p.preview,
    }));

  if (data.length === 0) return null;

  const explained = lsa.explained_variance_ratio ?? [];

  return {
    ...HC_DARK_THEME,
    chart: { ...HC_DARK_THEME.chart, type: "scatter", height: 320 },
    title: { text: undefined },
    xAxis: {
      ...HC_DARK_THEME.xAxis,
      title: {
        text: explained[0] !== undefined ? `LSA-1 (${(explained[0] * 100).toFixed(1)}%)` : "LSA-1",
        style: { color: "#64748b" },
      },
    },
    yAxis: {
      ...HC_DARK_THEME.yAxis,
      title: {
        text: explained[1] !== undefined ? `LSA-2 (${(explained[1] * 100).toFixed(1)}%)` : "LSA-2",
        style: { color: "#64748b" },
      },
    },
    tooltip: {
      useHTML: true,
      backgroundColor: "rgba(15,23,42,0.95)",
      borderColor: "rgba(255,255,255,0.1)",
      style: { color: "#e2e8f0" },
      formatter: function (this: any) {
        const p = this.point as { x?: number; y?: number; preview?: string };
        const xv = typeof p.x === "number" ? p.x.toFixed(3) : "—";
        const yv = typeof p.y === "number" ? p.y.toFixed(3) : "—";
        const text = (p.preview ?? "").replace(/</g, "&lt;");
        return `<div style="max-width:280px;font-size:11px"><strong>${xv}, ${yv}</strong><br/>${text}</div>`;
      },
    },
    plotOptions: {
      scatter: {
        marker: { radius: 3, lineWidth: 0, fillColor: "rgba(34,211,238,0.7)" },
      },
    },
    series: [
      {
        type: "scatter",
        name: "documents",
        data,
      } as any,
    ],
  };
}

function NGramBar({ rows }: { rows: NGramRow[] }) {
  if (!rows || rows.length === 0) {
    return <p className="text-sm text-slate-500">No terms.</p>;
  }
  const max = Math.max(...rows.map((r) => r.count), 1);
  return (
    <div className="space-y-1.5">
      {rows.slice(0, 12).map((row) => {
        const pct = (row.count / max) * 100;
        return (
          <div key={row.term} className="flex items-center gap-2">
            <span className="w-32 shrink-0 truncate text-xs text-slate-300">{row.term}</span>
            <div className="relative h-4 flex-1 overflow-hidden rounded-md bg-white/[0.04]">
              <div
                className="h-full rounded-md bg-gradient-to-r from-cyan-500/70 to-cyan-300/70"
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className="w-10 shrink-0 text-right font-mono text-xs text-slate-400">
              {row.count}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function PatternPills({ patterns }: { patterns?: ColumnProfile["patterns"] }) {
  if (!patterns?.counts) return <p className="text-sm text-slate-500">No patterns scanned.</p>;
  const totalRows = patterns.total_rows ?? 0;
  const entries = Object.entries(patterns.counts);
  if (entries.length === 0 || entries.every(([, count]) => count === 0)) {
    return <p className="text-sm text-slate-500">No regex pattern hits.</p>;
  }
  return (
    <div className="space-y-2">
      {entries.map(([name, count]) => {
        if (count === 0) return null;
        const pct = totalRows > 0 ? (count / totalRows) * 100 : 0;
        const examples = patterns.examples?.[name] ?? [];
        return (
          <div key={name} className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
            <div className="flex items-center justify-between">
              <span className="text-sm capitalize text-slate-200">{name}</span>
              <span className="font-mono text-xs text-cyan-200">
                {count} rows · {pct.toFixed(1)}%
              </span>
            </div>
            {examples.length > 0 ? (
              <div className="mt-1 flex flex-wrap gap-1">
                {examples.map((ex, i) => (
                  <Badge key={i} variant="muted" className="font-mono">
                    {ex}
                  </Badge>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function SentimentBar({ sent }: { sent?: ColumnProfile["sentiment_lite"] }) {
  if (!sent || (sent.n_tokens ?? 0) === 0) {
    return <p className="text-sm text-slate-500">Not enough tokens.</p>;
  }
  const pos = sent.positive_ratio ?? 0;
  const neg = sent.negative_ratio ?? 0;
  const polarity = sent.polarity ?? 0;
  const polColor = polarity > 0.005 ? "text-emerald-300" : polarity < -0.005 ? "text-rose-300" : "text-slate-300";
  const total = pos + neg;
  const posWidth = total > 0 ? (pos / total) * 100 : 50;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>positive · {(pos * 100).toFixed(2)}%</span>
        <span className={polColor}>polarity {polarity >= 0 ? "+" : ""}{polarity.toFixed(4)}</span>
        <span>negative · {(neg * 100).toFixed(2)}%</span>
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-white/[0.04]">
        <div className="h-full bg-emerald-400/70" style={{ width: `${posWidth}%` }} />
        <div className="h-full bg-rose-400/70" style={{ width: `${100 - posWidth}%` }} />
      </div>
      <p className="text-xs text-slate-500">
        Tokens scanned: <span className="font-mono">{sent.n_tokens?.toLocaleString()}</span>
      </p>
    </div>
  );
}

export function TextProfilePanel({ textProfile }: TextProfilePanelProps) {
  const detected = textProfile?.detected_columns ?? [];
  const profiles = textProfile?.profiles ?? {};
  const firstColumn = detected[0] ?? null;
  const [activeColumn, setActiveColumn] = useState<string | null>(firstColumn);

  if (!textProfile) return null;

  if (!textProfile.available) {
    return (
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Text & NLP profile</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">FREE TEXT</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            Per-column linguistic profile: tokens, n-grams, regex patterns, language, sentiment, LSA.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-6 text-sm text-slate-400">
            {textProfile.reason ?? textProfile.error ?? "No text-like columns detected."}
          </div>
        </CardContent>
      </Card>
    );
  }

  const selected = activeColumn && profiles[activeColumn] ? activeColumn : firstColumn;
  const profile = selected ? profiles[selected] : undefined;
  const lsaChart = profile ? buildLsaScatter(profile.lsa) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Text & NLP profile</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">
              {detected.length} text col{detected.length === 1 ? "" : "s"}
            </span>
          </CardTitle>
          <CardDescription className="flex flex-wrap items-center gap-2 text-slate-400">
            <span>stopwords:</span>
            <Badge variant="outline" className="border-white/10 text-slate-200">
              {textProfile.stopwords_source}
            </Badge>
            {detected.length > 1 ? (
              <>
                <span>·</span>
                <span>profiled:</span>
                {(textProfile.profiled_columns ?? []).map((c) => (
                  <Badge key={c} variant="outline" className="border-cyan-400/20 text-cyan-100">
                    {c}
                  </Badge>
                ))}
              </>
            ) : null}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5 p-6">
          {detected.length > 1 ? (
            <div className="flex flex-wrap gap-2">
              {detected.map((col) => (
                <button
                  key={col}
                  onClick={() => setActiveColumn(col)}
                  className={`rounded-xl border px-3 py-1 text-xs font-semibold transition ${
                    selected === col
                      ? "border-cyan-400/40 bg-cyan-500/10 text-cyan-100"
                      : "border-white/10 bg-white/[0.02] text-slate-300 hover:border-white/20"
                  }`}
                >
                  {col}
                </button>
              ))}
            </div>
          ) : null}

          {profile && profile.available ? (
            <>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
                <Stat label="Rows" value={profile.n_rows ?? 0} />
                <Stat label="Vocab size" value={profile.tokens?.vocab_size ?? 0} />
                <Stat
                  label="Lexical diversity"
                  value={fmt(profile.tokens?.lexical_diversity, 3)}
                />
                <Stat label="Avg length" value={fmt(profile.length_stats?.avg, 1)} />
                <Stat label="Avg sentence" value={fmt(profile.avg_sentence_length, 1)} />
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">Language</p>
                  <p className="mt-1 text-sm text-slate-100">{profile.language?.language ?? "?"}</p>
                  <p className="text-[11px] text-slate-500">
                    confidence {fmt(profile.language?.confidence, 2)}
                  </p>
                </div>
                <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">
                    Sentiment-lite
                  </p>
                  <SentimentBar sent={profile.sentiment_lite} />
                </div>
                <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
                  <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">
                    Length range
                  </p>
                  <p className="mt-1 font-mono text-sm text-slate-100">
                    {profile.length_stats?.min} – {profile.length_stats?.max}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    median {fmt(profile.length_stats?.median, 0)}
                  </p>
                </div>
              </div>

              <div className="grid gap-4 xl:grid-cols-3">
                <section className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                  <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                    Top unigrams
                  </p>
                  <NGramBar rows={profile.top_unigrams ?? []} />
                </section>
                <section className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                  <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                    Top bigrams
                  </p>
                  <NGramBar rows={profile.top_bigrams ?? []} />
                </section>
                <section className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
                  <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                    Pattern hits
                  </p>
                  <PatternPills patterns={profile.patterns} />
                </section>
              </div>

              {lsaChart ? (
                <section>
                  <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                    Document map · {profile.lsa?.method}
                  </p>
                  <div className="overflow-hidden rounded-2xl border border-white/5 bg-white/[0.02] p-2">
                    <HighchartsReact highcharts={Highcharts} options={lsaChart} />
                  </div>
                </section>
              ) : profile.lsa ? (
                <p className="text-xs text-slate-500">
                  LSA preview unavailable: {profile.lsa.reason ?? "unknown"}.
                </p>
              ) : null}
            </>
          ) : profile ? (
            <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-6 text-sm text-slate-400">
              {profile.reason ?? "No profile available."}
            </div>
          ) : null}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
      <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-sm text-slate-50">{value}</p>
    </div>
  );
}
