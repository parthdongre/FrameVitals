import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { DistributionTelemetry } from "@/data/mockTelemetry";

interface TelemetryCardProps {
  distribution?: DistributionTelemetry | null;
}

export function TelemetryCard({ distribution }: TelemetryCardProps) {
  const safeDistribution: DistributionTelemetry = distribution ?? {
    title: "Distribution",
    subtitle: "No distribution data available yet.",
    points: [],
    mean: 0,
    stdDev: 0,
    min: 0,
    max: 0,
    totalRows: 0,
  };
  const safePoints = Array.isArray(safeDistribution.points) ? safeDistribution.points : [];

  const options: Highcharts.Options = {
    chart: {
      backgroundColor: "transparent",
      height: 330,
      style: {
        fontFamily: 'Geist, Satoshi, Inter, system-ui, sans-serif',
      },
      spacing: [8, 0, 0, 0],
    },
    title: { text: undefined },
    credits: { enabled: false },
    legend: { enabled: false },
    colors: ["#06b6d4"],
    xAxis: {
      categories: safePoints.map((point) => point.label),
      lineColor: "rgba(148, 163, 184, 0.0)",
      lineWidth: 0,
      tickColor: "rgba(148, 163, 184, 0.0)",
      tickLength: 0,
      gridLineWidth: 0,
      labels: {
        style: {
          color: "#94a3b8",
          fontSize: "11px",
          fontWeight: "500",
        },
      },
    },
    yAxis: {
      title: { text: undefined },
      gridLineWidth: 0,
      minorGridLineWidth: 0,
      lineWidth: 0,
      labels: {
        style: {
          color: "#64748b",
          fontSize: "11px",
        },
      },
    },
    tooltip: {
      backgroundColor: "rgba(2, 6, 23, 0.96)",
      borderColor: "rgba(6, 182, 212, 0.35)",
      borderRadius: 14,
      style: {
        color: "#e2e8f0",
      },
      useHTML: false,
    },
    plotOptions: {
      column: {
        borderWidth: 0,
        borderRadius: 7,
        groupPadding: 0.08,
        pointPadding: 0.04,
        shadow: {
          color: "rgba(6, 182, 212, 0.28)",
          offsetX: 0,
          offsetY: 0,
          opacity: 0.9,
          width: 10,
        },
      },
      series: {
        animation: {
          duration: 900,
        },
        states: {
          inactive: {
            opacity: 0.6,
          },
        },
      },
    },
    series: [
      {
        type: "column",
        name: safeDistribution.title,
        data: safePoints.map((point) => point.value),
        color: {
          linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
          stops: [
            [0, "#22d3ee"],
            [1, "#06b6d4"],
          ],
        },
      },
    ],
  };

  return (
    <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
      <CardHeader className="border-b border-white/5 bg-white/[0.015]">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
              <CardTitle className="text-slate-50">{safeDistribution.title}</CardTitle>
            <CardDescription className="mt-1 max-w-2xl text-slate-400">
                {safeDistribution.subtitle}
            </CardDescription>
          </div>
          <Badge variant="outline" className="border-cyan-400/20 text-cyan-200">
              Distribution
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-5 p-6">
        <div className="rounded-[1.45rem] border border-white/5 bg-space-900/60 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
          <HighchartsReact highcharts={Highcharts} options={options} />
        </div>

        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <SummaryTile label="Mean" value={safeDistribution.mean.toFixed(1)} />
          <SummaryTile label="Std Dev" value={safeDistribution.stdDev.toFixed(1)} />
          <SummaryTile label="Min" value={safeDistribution.min.toFixed(1)} />
          <SummaryTile label="Max" value={safeDistribution.max.toFixed(1)} />
        </div>

        <div className="rounded-2xl border border-white/5 bg-white/[0.02] px-4 py-3 text-sm text-slate-400">
          <span className="font-mono text-cyan-200">{safeDistribution.totalRows.toLocaleString()}</span>
          <span className="ml-2">rows instrumented across the current scan window.</span>
        </div>
      </CardContent>
    </Card>
  );
}

function SummaryTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
      <p className="text-[11px] uppercase tracking-[0.35em] text-slate-500">{label}</p>
      <p className="mt-2 font-mono text-2xl text-slate-50">{value}</p>
    </div>
  );
}