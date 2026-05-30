import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import "@/charts/theme";
import { ChartFrame } from "@/components/ui/ChartFrame";
import { EmptyState } from "@/components/ui/EmptyState";
import { safeArr, safeNum, safeObj, safeStr } from "@/lib/safe";
import type { DistributionTelemetry } from "@/data/payload";

interface DistributionHistogramProps {
  distribution: unknown;
  className?: string;
  /**
   * Optional override; defaults to the distribution's own title.
   */
  title?: string;
}

/**
 * Histogram backed by the precomputed `distribution.points` payload from
 * `modules/frontend_api.py`. The shape always carries label + value pairs,
 * so we render a column chart with the bins on the x-axis.
 */
export function DistributionHistogram({
  distribution,
  className,
  title,
}: DistributionHistogramProps) {
  const dist = safeObj(distribution, {} as Partial<DistributionTelemetry>);
  const points = safeArr<{ label?: string; value?: number }>(dist.points);

  if (!points.length) {
    return (
      <ChartFrame
        eyebrow="Distribution"
        title={title ?? safeStr(dist.title, "Feature distribution")}
        description={safeStr(dist.subtitle, "Histogram of observed values.")}
        className={className}
      >
        <EmptyState
          compact
          title="Distribution unavailable"
          hint="No usable values were detected in the primary feature column."
        />
      </ChartFrame>
    );
  }

  const categories = points.map((p) => safeStr(p.label, ""));
  const data = points.map((p) => safeNum(p.value));

  const options: Highcharts.Options = {
    chart: { type: "column", height: 280 },
    title: { text: undefined },
    xAxis: { categories, labels: { rotation: -25, step: Math.ceil(categories.length / 12) || 1 } },
    yAxis: { title: { text: "Count" }, allowDecimals: false },
    tooltip: { headerFormat: "<span class=\"label-mono\">{point.key}</span><br/>", pointFormat: "<b>{point.y}</b> rows" },
    legend: { enabled: false },
    plotOptions: {
      column: {
        pointPadding: 0.05,
        borderWidth: 0,
        groupPadding: 0,
      },
    },
    series: [
      {
        type: "column",
        name: "Count",
        data,
        color: "var(--accent)",
      } as Highcharts.SeriesColumnOptions,
    ],
  };

  return (
    <ChartFrame
      eyebrow="Distribution"
      title={title ?? safeStr(dist.title, "Feature distribution")}
      description={safeStr(dist.subtitle, "Histogram of observed values.")}
      className={className}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 px-1 font-mono text-[11px] tabular-nums text-ink-3 sm:grid-cols-4">
        {[
          ["mean", dist.mean],
          ["std dev", dist.stdDev],
          ["min", dist.min],
          ["max", dist.max],
        ].map(([k, v]) => (
          <div key={String(k)} className="flex items-baseline justify-between gap-2">
            <dt className="uppercase tracking-[0.2em] text-ink-4">{String(k)}</dt>
            <dd className="text-ink-2">{Number.isFinite(v as number) ? Number(v).toFixed(2) : "—"}</dd>
          </div>
        ))}
      </dl>
    </ChartFrame>
  );
}
