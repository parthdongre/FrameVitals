import { motion } from "framer-motion";
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface TimeSeriesPanelProps {
  timeSeries?: {
    available?: boolean;
    reason?: string;
    error?: string;
    detected_date_column?: string;
    date_score?: number;
    numeric_column?: string;
    n_observations?: number;
    date_range?: { start: string; end: string };
    frequency?: {
      available?: boolean;
      label?: string;
      pandas_inferred?: string | null;
      median_seconds?: number;
      n_samples?: number;
    };
    period_estimate?: number | null;
    stationarity?: {
      available?: boolean;
      adf?: { p_value?: number };
      kpss?: { p_value?: number };
      verdict?: string;
    };
    decomposition?: {
      available?: boolean;
      reason?: string;
      period?: number;
      trend_strength?: number;
      seasonal_strength?: number;
      trend_summary?: { first?: number; last?: number; delta?: number };
      trend_preview?: number[];
      seasonal_preview?: number[];
      residual_preview?: number[];
    };
    autocorrelation?: {
      available?: boolean;
      max_lag?: number;
      acf?: number[];
      pacf?: number[];
    };
    forecast?: {
      available?: boolean;
      reason?: string;
      method?: string;
      horizon?: number;
      seasonal_periods?: number | null;
      holt_winters?: { metrics?: { mae?: number; rmse?: number }; preview?: number[] };
      naive_baseline?: { metrics?: { mae?: number; rmse?: number }; preview?: number[] };
      actual_preview?: number[];
    };
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

function verdictTone(verdict?: string): "high" | "medium" | "low" | "muted" {
  if (!verdict) return "muted";
  if (verdict.startsWith("stationary")) return "high";
  if (verdict.includes("trend") || verdict.includes("difference")) return "medium";
  if (verdict.includes("non-stationary")) return "low";
  return "muted";
}

const HC_DARK_THEME: Highcharts.Options = {
  chart: {
    backgroundColor: "transparent",
    style: { fontFamily: "Inter, system-ui, sans-serif" },
  },
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
  plotOptions: {
    series: { animation: { duration: 600 } },
    line: { lineWidth: 2 },
  },
};

function buildDecompositionChart(d: NonNullable<TimeSeriesPanelProps["timeSeries"]>["decomposition"]) {
  if (!d?.available) return null;
  const trend = d.trend_preview ?? [];
  const seasonal = d.seasonal_preview ?? [];
  const resid = d.residual_preview ?? [];
  const length = Math.max(trend.length, seasonal.length, resid.length);
  if (length === 0) return null;

  const options: Highcharts.Options = {
    ...HC_DARK_THEME,
    chart: { ...HC_DARK_THEME.chart, height: 260, type: "line" },
    title: { text: undefined },
    xAxis: { ...HC_DARK_THEME.xAxis, title: { text: "Last 50 observations", style: { color: "#64748b" } } },
    series: [
      {
        type: "line",
        name: "Trend",
        data: trend,
        color: "#22d3ee",
      },
      {
        type: "line",
        name: "Seasonal",
        data: seasonal,
        color: "#a78bfa",
      },
      {
        type: "line",
        name: "Residual",
        data: resid,
        color: "#94a3b8",
        dashStyle: "ShortDot",
      },
    ],
  };
  return options;
}

function buildAutocorrChart(a: NonNullable<TimeSeriesPanelProps["timeSeries"]>["autocorrelation"]) {
  if (!a?.available) return null;
  const acf = a.acf ?? [];
  const pacf = a.pacf ?? [];
  if (acf.length === 0 && pacf.length === 0) return null;

  const options: Highcharts.Options = {
    ...HC_DARK_THEME,
    chart: { ...HC_DARK_THEME.chart, height: 260, type: "column" },
    title: { text: undefined },
    xAxis: { ...HC_DARK_THEME.xAxis, title: { text: "Lag", style: { color: "#64748b" } } },
    yAxis: {
      ...HC_DARK_THEME.yAxis,
      plotLines: [
        { value: 0, color: "rgba(255,255,255,0.2)", width: 1 },
      ],
    },
    plotOptions: {
      column: { groupPadding: 0.05, pointPadding: 0.05, borderWidth: 0 },
    },
    series: [
      { type: "column", name: "ACF", data: acf, color: "#22d3ee" },
      { type: "column", name: "PACF", data: pacf, color: "#a78bfa" },
    ],
  };
  return options;
}

function buildForecastChart(f: NonNullable<TimeSeriesPanelProps["timeSeries"]>["forecast"]) {
  if (!f?.available) return null;
  const actual = f.actual_preview ?? [];
  const hw = f.holt_winters?.preview ?? [];
  const naive = f.naive_baseline?.preview ?? [];
  if (actual.length === 0 && hw.length === 0) return null;

  const options: Highcharts.Options = {
    ...HC_DARK_THEME,
    chart: { ...HC_DARK_THEME.chart, height: 260, type: "line" },
    title: { text: undefined },
    xAxis: { ...HC_DARK_THEME.xAxis, title: { text: "Forecast horizon", style: { color: "#64748b" } } },
    series: [
      { type: "line", name: "Actual", data: actual, color: "#22d3ee" },
      { type: "line", name: "Holt-Winters", data: hw, color: "#a78bfa", dashStyle: "Dash" },
      { type: "line", name: "Naive baseline", data: naive, color: "#94a3b8", dashStyle: "ShortDot" },
    ],
  };
  return options;
}

export function TimeSeriesPanel({ timeSeries }: TimeSeriesPanelProps) {
  if (!timeSeries) return null;

  if (!timeSeries.available) {
    return (
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Time-series analysis</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">AUTO</span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            STL decomposition, stationarity tests, ACF/PACF and Holt-Winters baseline forecast.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-6">
          <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-6 text-sm text-slate-400">
            {timeSeries.reason ?? timeSeries.error ?? "No date-like column detected."}
          </div>
        </CardContent>
      </Card>
    );
  }

  const decomp = timeSeries.decomposition;
  const acfChart = buildAutocorrChart(timeSeries.autocorrelation);
  const decompChart = buildDecompositionChart(timeSeries.decomposition);
  const forecastChart = buildForecastChart(timeSeries.forecast);
  const verdict = timeSeries.stationarity?.verdict;
  const tone = verdictTone(verdict);

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Time-series analysis</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">AUTO</span>
          </CardTitle>
          <CardDescription className="flex flex-wrap items-center gap-2 text-slate-400">
            <span>date column:</span>
            <Badge variant="outline" className="border-cyan-400/20 text-cyan-100">
              {timeSeries.detected_date_column}
            </Badge>
            <span>·</span>
            <span>series:</span>
            <Badge variant="outline" className="border-violet-400/20 text-violet-100">
              {timeSeries.numeric_column}
            </Badge>
            <span>·</span>
            <span>{timeSeries.n_observations ?? 0} observations</span>
            <span>·</span>
            <span>frequency: <span className="font-mono text-cyan-200">{timeSeries.frequency?.label ?? "?"}</span></span>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6 p-6">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Range start" value={timeSeries.date_range?.start ?? "—"} mono />
            <Stat label="Range end" value={timeSeries.date_range?.end ?? "—"} mono />
            <Stat label="Period (FFT)" value={fmt(timeSeries.period_estimate)} />
            <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
              <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">
                Stationarity verdict
              </p>
              <div className="mt-1">
                <Badge
                  variant={
                    tone === "high" ? "cyan" : tone === "medium" ? "violet" : "muted"
                  }
                >
                  {verdict ?? "—"}
                </Badge>
              </div>
              {timeSeries.stationarity?.available ? (
                <p className="mt-2 text-[11px] text-slate-500">
                  ADF p={fmt(timeSeries.stationarity.adf?.p_value)} · KPSS p=
                  {fmt(timeSeries.stationarity.kpss?.p_value)}
                </p>
              ) : null}
            </div>
          </div>

          {decomp?.available ? (
            <section>
              <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                STL decomposition (period {decomp.period})
              </p>
              <div className="grid gap-4 xl:grid-cols-3">
                <Stat label="Trend strength" value={fmt(decomp.trend_strength, 3)} />
                <Stat label="Seasonal strength" value={fmt(decomp.seasonal_strength, 3)} />
                <Stat
                  label="Trend Δ (last - first)"
                  value={fmt(decomp.trend_summary?.delta)}
                />
              </div>
              {decompChart ? (
                <div className="mt-3 overflow-hidden rounded-2xl border border-white/5 bg-white/[0.02] p-2">
                  <HighchartsReact highcharts={Highcharts} options={decompChart} />
                </div>
              ) : null}
            </section>
          ) : (
            <p className="text-xs text-slate-500">
              STL decomposition unavailable: {decomp?.reason ?? "no period detected"}.
            </p>
          )}

          {acfChart ? (
            <section>
              <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                Autocorrelation (max lag {timeSeries.autocorrelation?.max_lag})
              </p>
              <div className="overflow-hidden rounded-2xl border border-white/5 bg-white/[0.02] p-2">
                <HighchartsReact highcharts={Highcharts} options={acfChart} />
              </div>
            </section>
          ) : null}

          {forecastChart ? (
            <section>
              <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                Forecast preview · {timeSeries.forecast?.method}
              </p>
              <div className="grid gap-2 sm:grid-cols-4">
                <Stat
                  label="HW MAE"
                  value={fmt(timeSeries.forecast?.holt_winters?.metrics?.mae)}
                />
                <Stat
                  label="HW RMSE"
                  value={fmt(timeSeries.forecast?.holt_winters?.metrics?.rmse)}
                />
                <Stat
                  label="Naive MAE"
                  value={fmt(timeSeries.forecast?.naive_baseline?.metrics?.mae)}
                />
                <Stat
                  label="Naive RMSE"
                  value={fmt(timeSeries.forecast?.naive_baseline?.metrics?.rmse)}
                />
              </div>
              <div className="mt-3 overflow-hidden rounded-2xl border border-white/5 bg-white/[0.02] p-2">
                <HighchartsReact highcharts={Highcharts} options={forecastChart} />
              </div>
            </section>
          ) : timeSeries.forecast ? (
            <p className="text-xs text-slate-500">
              Forecast unavailable: {timeSeries.forecast.reason ?? "n/a"}.
            </p>
          ) : null}
        </CardContent>
      </Card>
    </motion.div>
  );
}

function Stat({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
      <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">{label}</p>
      <p className={`mt-1 ${mono ? "font-mono" : ""} text-sm text-slate-50`}>{value}</p>
    </div>
  );
}
