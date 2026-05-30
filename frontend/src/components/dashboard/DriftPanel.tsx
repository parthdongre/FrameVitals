import { useRef, useState } from "react";
import { motion } from "framer-motion";
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type Severity = "stable" | "minor" | "moderate" | "severe" | "unknown";

interface NumericColumn {
  column: string;
  type: "numeric";
  available: boolean;
  reason?: string;
  psi?: number;
  psi_severity?: Severity;
  ks_p_value?: number;
  ks_significant?: boolean;
  ref_mean?: number;
  cur_mean?: number;
  z_shift?: number;
  histogram?: { edges: number[]; ref: number[]; cur: number[] };
}

interface CategoricalColumn {
  column: string;
  type: "categorical";
  available: boolean;
  reason?: string;
  psi?: number;
  psi_severity?: Severity;
  chi2_p_value?: number;
  chi2_significant?: boolean;
  new_categories?: string[];
  missing_categories?: string[];
  distribution?: { categories: string[]; ref_props: number[]; cur_props: number[] };
}

type ColumnResult = NumericColumn | CategoricalColumn | {
  column: string;
  type: "unsupported";
  available: false;
  reason?: string;
};

interface DriftReport {
  available?: boolean;
  reason?: string;
  reference_filename?: string;
  current_filename?: string;
  ref_shape?: number[];
  cur_shape?: number[];
  shared_columns?: string[];
  summary?: {
    n_columns_compared: number;
    severity_counts: Record<Severity, number>;
    overall_verdict: Severity;
  };
  interpretation?: string;
  columns?: ColumnResult[];
  split_by?: string;
  split_ratio?: number;
}

function fmt(value: unknown, digits = 4): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    return value.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return String(value);
}

const HC_THEME: Highcharts.Options = {
  chart: { backgroundColor: "transparent", style: { fontFamily: "Inter, system-ui, sans-serif" } },
  credits: { enabled: false },
  legend: { itemStyle: { color: "#cbd5e1" }, padding: 4 },
  xAxis: {
    labels: { style: { color: "#94a3b8", fontSize: "10px" } },
    lineColor: "rgba(255,255,255,0.05)",
    tickColor: "rgba(255,255,255,0.05)",
  },
  yAxis: {
    labels: { style: { color: "#94a3b8", fontSize: "10px" } },
    gridLineColor: "rgba(255,255,255,0.04)",
    title: { text: undefined },
  },
};

function severityClass(severity?: Severity): string {
  switch (severity) {
    case "severe":
      return "border-rose-400/30 bg-rose-500/15 text-rose-200";
    case "moderate":
      return "border-amber-400/30 bg-amber-500/15 text-amber-200";
    case "minor":
      return "border-cyan-400/30 bg-cyan-500/15 text-cyan-100";
    case "stable":
      return "border-emerald-400/30 bg-emerald-500/15 text-emerald-200";
    default:
      return "border-white/10 bg-white/[0.04] text-slate-300";
  }
}

function buildNumericChart(col: NumericColumn): Highcharts.Options | null {
  if (!col.histogram) return null;
  const { edges, ref, cur } = col.histogram;
  if (edges.length < 2) return null;
  const labels = edges.slice(0, -1).map((edge, i) => {
    const right = edges[i + 1];
    const a = typeof edge === "number" ? edge.toFixed(2) : String(edge);
    const b = typeof right === "number" ? right.toFixed(2) : String(right);
    return `${a}-${b}`;
  });
  return {
    ...HC_THEME,
    chart: { ...HC_THEME.chart, type: "column", height: 220 },
    title: { text: undefined },
    xAxis: { ...HC_THEME.xAxis, categories: labels, tickInterval: 4 },
    plotOptions: { column: { groupPadding: 0.05, pointPadding: 0.02, borderWidth: 0 } },
    series: [
      { type: "column", name: "Reference", data: ref, color: "rgba(34,211,238,0.6)" },
      { type: "column", name: "Current", data: cur, color: "rgba(167,139,250,0.7)" },
    ],
  };
}

function buildCategoricalChart(col: CategoricalColumn): Highcharts.Options | null {
  if (!col.distribution) return null;
  const { categories, ref_props, cur_props } = col.distribution;
  if (categories.length === 0) return null;
  return {
    ...HC_THEME,
    chart: { ...HC_THEME.chart, type: "bar", height: Math.max(220, categories.length * 22) },
    title: { text: undefined },
    xAxis: { ...HC_THEME.xAxis, categories },
    yAxis: {
      ...HC_THEME.yAxis,
      labels: { style: { color: "#94a3b8", fontSize: "10px" }, formatter: function () { return `${(Number(this.value) * 100).toFixed(0)}%`; } },
    },
    plotOptions: { bar: { groupPadding: 0.05, pointPadding: 0.02, borderWidth: 0 } },
    series: [
      { type: "bar", name: "Reference", data: ref_props, color: "rgba(34,211,238,0.6)" },
      { type: "bar", name: "Current", data: cur_props, color: "rgba(167,139,250,0.7)" },
    ],
  };
}

export function DriftPanel() {
  const [report, setReport] = useState<DriftReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"two-files" | "split-by-date">("two-files");
  const [dateColumn, setDateColumn] = useState("");
  const [ratio, setRatio] = useState("0.5");
  const [activeColumn, setActiveColumn] = useState<string | null>(null);

  const referenceRef = useRef<HTMLInputElement>(null);
  const currentRef = useRef<HTMLInputElement>(null);
  const datasetRef = useRef<HTMLInputElement>(null);

  const submit = async () => {
    setLoading(true);
    setError(null);
    setReport(null);
    setActiveColumn(null);

    try {
      let response: Response;
      if (mode === "two-files") {
        const ref = referenceRef.current?.files?.[0];
        const cur = currentRef.current?.files?.[0];
        if (!ref || !cur) {
          throw new Error("Select both a reference and a current dataset.");
        }
        const fd = new FormData();
        fd.append("reference", ref);
        fd.append("current", cur);
        response = await fetch("/api/compare", { method: "POST", body: fd });
      } else {
        const ds = datasetRef.current?.files?.[0];
        if (!ds) throw new Error("Select a dataset to split.");
        if (!dateColumn.trim()) throw new Error("Provide a date column name to split on.");
        const fd = new FormData();
        fd.append("dataset", ds);
        fd.append("date_column", dateColumn.trim());
        fd.append("ratio", ratio);
        response = await fetch("/api/compare-self", { method: "POST", body: fd });
      }

      const body = await response.json();
      if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
      setReport(body);
      const firstAvailable = (body.columns ?? []).find((c: ColumnResult) => c.available);
      if (firstAvailable) setActiveColumn(firstAvailable.column);
    } catch (exc: any) {
      setError(exc?.message ?? String(exc));
    } finally {
      setLoading(false);
    }
  };

  const summary = report?.summary;
  const columns = report?.columns ?? [];
  const selected = columns.find((c) => c.column === activeColumn) ?? columns.find((c) => c.available) ?? null;

  let detailChart: Highcharts.Options | null = null;
  if (selected && selected.available) {
    if (selected.type === "numeric") detailChart = buildNumericChart(selected as NumericColumn);
    else if (selected.type === "categorical") detailChart = buildCategoricalChart(selected as CategoricalColumn);
  }

  return (
    <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45 }}>
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Drift / compare mode</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">PSI · KS · χ²</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            Compare two datasets, or split one by a date column, and quantify how much each column has shifted.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5 p-6">
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setMode("two-files")}
              className={`rounded-xl border px-3 py-1 text-xs font-semibold transition ${
                mode === "two-files"
                  ? "border-cyan-400/40 bg-cyan-500/10 text-cyan-100"
                  : "border-white/10 bg-white/[0.02] text-slate-300 hover:border-white/20"
              }`}
            >
              Two files
            </button>
            <button
              onClick={() => setMode("split-by-date")}
              className={`rounded-xl border px-3 py-1 text-xs font-semibold transition ${
                mode === "split-by-date"
                  ? "border-cyan-400/40 bg-cyan-500/10 text-cyan-100"
                  : "border-white/10 bg-white/[0.02] text-slate-300 hover:border-white/20"
              }`}
            >
              Split one file by date
            </button>
          </div>

          {mode === "two-files" ? (
            <div className="grid gap-3 md:grid-cols-2">
              <FileBox label="Reference (older / training)" inputRef={referenceRef} />
              <FileBox label="Current (newer / production)" inputRef={currentRef} />
            </div>
          ) : (
            <div className="grid gap-3 md:grid-cols-3">
              <FileBox label="Dataset" inputRef={datasetRef} className="md:col-span-3" />
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">Date column</p>
                <input
                  value={dateColumn}
                  onChange={(e) => setDateColumn(e.target.value)}
                  placeholder="e.g. timestamp"
                  className="mt-1 w-full rounded-md border border-white/5 bg-white/[0.03] px-3 py-1.5 text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                />
              </div>
              <div className="rounded-xl border border-white/5 bg-white/[0.02] p-3">
                <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">Older fraction (0.1–0.9)</p>
                <input
                  value={ratio}
                  onChange={(e) => setRatio(e.target.value)}
                  className="mt-1 w-full rounded-md border border-white/5 bg-white/[0.03] px-3 py-1.5 font-mono text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                />
              </div>
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={submit}
              disabled={loading}
              className="rounded-2xl border border-cyan-400/15 bg-cyan-500 px-4 py-2 text-sm font-semibold text-space-950 transition hover:bg-cyan-400 disabled:opacity-40"
            >
              {loading ? "Comparing…" : "Run drift comparison"}
            </button>
            {error ? <span className="text-xs text-rose-300">{error}</span> : null}
          </div>

          {report?.available ? (
            <>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
                <Stat label="Reference" value={`${report.ref_shape?.[0] ?? 0} × ${report.ref_shape?.[1] ?? 0}`} />
                <Stat label="Current" value={`${report.cur_shape?.[0] ?? 0} × ${report.cur_shape?.[1] ?? 0}`} />
                <Stat label="Columns compared" value={summary?.n_columns_compared ?? 0} />
                <Stat label="Severe" value={summary?.severity_counts?.severe ?? 0} />
                <Stat label="Moderate" value={summary?.severity_counts?.moderate ?? 0} />
                <div className={`rounded-xl border px-3 py-2 ${severityClass(summary?.overall_verdict)}`}>
                  <p className="text-[10px] uppercase tracking-[0.28em] opacity-70">Overall</p>
                  <p className="mt-1 font-mono text-sm">{summary?.overall_verdict ?? "—"}</p>
                </div>
              </div>

              <div className="overflow-x-auto rounded-2xl border border-white/5 bg-white/[0.02]">
                <table className="w-full border-separate border-spacing-0 text-left text-sm">
                  <thead className="bg-white/[0.02] text-[11px] uppercase tracking-[0.3em] text-slate-500">
                    <tr>
                      <th className="border-b border-white/5 px-3 py-2">Column</th>
                      <th className="border-b border-white/5 px-3 py-2">Type</th>
                      <th className="border-b border-white/5 px-3 py-2 text-right">PSI</th>
                      <th className="border-b border-white/5 px-3 py-2">Severity</th>
                      <th className="border-b border-white/5 px-3 py-2 text-right">Test p-value</th>
                      <th className="border-b border-white/5 px-3 py-2 text-right">Mean shift (z)</th>
                      <th className="border-b border-white/5 px-3 py-2">Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {columns.map((col, idx) => {
                      const sev = (col.available ? col.psi_severity : "unknown") ?? "unknown";
                      const isActive = selected?.column === col.column;
                      return (
                        <tr
                          key={col.column}
                          onClick={() => col.available && setActiveColumn(col.column)}
                          className={`border-b border-white/5 cursor-pointer transition ${
                            isActive ? "bg-cyan-500/[0.06]" : idx % 2 ? "" : "bg-white/[0.01]"
                          } hover:bg-white/[0.04]`}
                        >
                          <td className="border-b border-white/5 px-3 py-2 text-slate-100">{col.column}</td>
                          <td className="border-b border-white/5 px-3 py-2 text-slate-400">{col.type}</td>
                          <td className="border-b border-white/5 px-3 py-2 text-right font-mono text-slate-200">
                            {col.available ? fmt(col.psi) : "—"}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2">
                            <span className={`inline-block rounded-full border px-2 py-0.5 text-[11px] ${severityClass(sev)}`}>
                              {sev}
                            </span>
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 text-right font-mono text-slate-400">
                            {col.available
                              ? col.type === "numeric"
                                ? fmt((col as NumericColumn).ks_p_value)
                                : fmt((col as CategoricalColumn).chi2_p_value)
                              : "—"}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 text-right font-mono text-slate-400">
                            {col.available && col.type === "numeric" ? fmt((col as NumericColumn).z_shift, 3) : "—"}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 text-slate-400">
                            {col.available
                              ? col.type === "categorical" && (col as CategoricalColumn).new_categories?.length
                                ? `new: ${(col as CategoricalColumn).new_categories!.slice(0, 3).join(", ")}`
                                : ""
                              : col.reason ?? "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {selected && selected.available ? (
                <section>
                  <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                    Distribution overlay · {selected.column}
                  </p>
                  {detailChart ? (
                    <div className="overflow-hidden rounded-2xl border border-white/5 bg-white/[0.02] p-2">
                      <HighchartsReact highcharts={Highcharts} options={detailChart} />
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500">No distribution preview for this column.</p>
                  )}
                </section>
              ) : null}

              {report.interpretation ? (
                <p className="text-xs text-slate-500">{report.interpretation}</p>
              ) : null}
            </>
          ) : report ? (
            <p className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-4 text-sm text-slate-400">
              {report.reason ?? "Drift report unavailable."}
            </p>
          ) : null}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function FileBox({
  label,
  inputRef,
  className = "",
}: {
  label: string;
  inputRef: React.RefObject<HTMLInputElement | null>;
  className?: string;
}) {
  const [name, setName] = useState<string | null>(null);
  return (
    <div className={`rounded-xl border border-white/10 bg-white/[0.02] p-3 ${className}`}>
      <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">{label}</p>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv,.xlsx,.xls,.json"
        className="mt-2 block w-full text-xs text-slate-300 file:mr-3 file:rounded-md file:border file:border-white/10 file:bg-white/[0.04] file:px-3 file:py-1 file:text-cyan-100 hover:file:bg-white/[0.08]"
        onChange={(e) => setName(e.target.files?.[0]?.name ?? null)}
      />
      {name ? <p className="mt-1 truncate font-mono text-xs text-slate-400">{name}</p> : null}
    </div>
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
