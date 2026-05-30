import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import "@/charts/theme";
import { ChartFrame } from "@/components/ui/ChartFrame";
import { EmptyState } from "@/components/ui/EmptyState";
import { safeNum, safeObj, isPresent } from "@/lib/safe";
import { formatLabel } from "@/lib/format";

interface HealthRadarProps {
  /**
   * Either a `health.components` object (component → score 0-100) or the full
   * health payload from the backend.
   */
  health: unknown;
  className?: string;
}

/**
 * Health components rendered as a polar bar / radar chart. Single-series so
 * the visual stays calm; the y-axis is fixed to 0..100 so cross-dataset
 * comparisons land at the same scale.
 */
export function HealthRadar({ health, className }: HealthRadarProps) {
  const root = safeObj(health, {} as Record<string, unknown>);
  const components = safeObj(
    (root.components ?? root) as unknown,
    {} as Record<string, unknown>,
  );

  const entries = Object.entries(components)
    .map(([k, v]) => {
      // Components can be either { score: number } or a bare number.
      const num =
        typeof v === "number"
          ? v
          : safeNum((v as { score?: unknown })?.score, NaN);
      return [k, num] as const;
    })
    .filter(([, v]) => Number.isFinite(v) && v > 0);

  if (!isPresent(components) || entries.length === 0) {
    return (
      <ChartFrame
        eyebrow="Health"
        title="Component scores"
        description="Composite breakdown of the dataset's health score."
        className={className}
      >
        <EmptyState
          compact
          title="Health components unavailable"
          hint="The backend did not return a per-component breakdown for this dataset."
        />
      </ChartFrame>
    );
  }

  const categories = entries.map(([k]) => formatLabel(k));
  const data = entries.map(([, v]) => Number(v.toFixed(1)));

  const options: Highcharts.Options = {
    chart: { polar: true, type: "column", height: 320 },
    title: { text: undefined },
    xAxis: {
      categories,
      tickmarkPlacement: "on",
      lineWidth: 0,
      labels: { distance: 14 },
    },
    yAxis: {
      min: 0,
      max: 100,
      gridLineInterpolation: "polygon",
      tickInterval: 20,
      labels: { format: "{value}" },
    },
    pane: { size: "84%" },
    legend: { enabled: false },
    tooltip: { pointFormat: "<b>{point.y:.1f}</b> / 100" },
    plotOptions: {
      column: {
        pointPadding: 0,
        groupPadding: 0,
        borderWidth: 0,
        colorByPoint: true,
      },
    },
    series: [
      {
        type: "column",
        name: "Health",
        data,
      },
    ],
  };

  return (
    <ChartFrame
      eyebrow="Health"
      title="Component scores"
      description="Composite breakdown of the dataset's health score."
      className={className}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </ChartFrame>
  );
}
