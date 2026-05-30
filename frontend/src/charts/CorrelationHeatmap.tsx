import { useMemo } from "react";
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import HeatmapModule from "highcharts/modules/heatmap";
import "@/charts/theme";
import { ChartFrame } from "@/components/ui/ChartFrame";
import { EmptyState } from "@/components/ui/EmptyState";
import { safeObj } from "@/lib/safe";

// Highcharts 12 ships modules as direct ES imports that auto-register;
// the legacy callable form is preserved here for older versions.
if (typeof (HeatmapModule as unknown as (h: typeof Highcharts) => void) === "function") {
  try {
    (HeatmapModule as unknown as (h: typeof Highcharts) => void)(Highcharts);
  } catch {
    /* already registered */
  }
}

interface CorrelationHeatmapProps {
  /**
   * `profile.correlations`: nested object {colA: {colB: number, ...}, ...}.
   */
  correlations: unknown;
  className?: string;
  /**
   * Limit to this many rows/columns (selected by absolute correlation sum).
   */
  limit?: number;
}

/**
 * Pearson correlation heatmap rendered with a teal divergent scale.
 *
 * The diagonal is dimmed so the off-diagonal pattern is the focus. Numeric
 * coordinates use `[xIndex, yIndex, value]` per Highcharts heatmap convention.
 */
export function CorrelationHeatmap({
  correlations,
  className,
  limit = 16,
}: CorrelationHeatmapProps) {
  const matrix = safeObj(correlations, {} as Record<string, Record<string, unknown>>);
  const allCols = Object.keys(matrix);

  const { columns, data } = useMemo(() => {
    if (allCols.length === 0)
      return { columns: [] as string[], data: [] as Array<[number, number, number]> };

    // Rank columns by their summed absolute correlations to keep the
    // densest part of the matrix when truncating.
    const sumAbs = (col: string) =>
      Object.values(matrix[col] ?? {}).reduce(
        (acc: number, v) => acc + Math.abs(typeof v === "number" ? v : 0),
        0,
      );
    const ranked = [...allCols].sort((a, b) => sumAbs(b) - sumAbs(a));
    const cols = ranked.slice(0, limit);

    const points: Array<[number, number, number]> = [];
    cols.forEach((rowCol, y) => {
      cols.forEach((colCol, x) => {
        const raw = matrix[rowCol]?.[colCol];
        const value = typeof raw === "number" && Number.isFinite(raw) ? raw : 0;
        points.push([x, y, Number(value.toFixed(3))]);
      });
    });

    return { columns: cols, data: points };
  }, [matrix, allCols, limit]);

  if (columns.length === 0) {
    return (
      <ChartFrame
        eyebrow="Correlation"
        title="Pearson correlation"
        description="Numeric column pairs ranked by absolute correlation."
        className={className}
      >
        <EmptyState
          compact
          title="No correlations available"
          hint="The dataset doesn't contain enough numeric columns for a correlation matrix."
        />
      </ChartFrame>
    );
  }

  const options: Highcharts.Options = {
    chart: {
      type: "heatmap",
      height: Math.max(260, columns.length * 22 + 80),
    },
    title: { text: undefined },
    xAxis: { categories: columns, labels: { rotation: -35, style: { fontSize: "10px" } } },
    yAxis: {
      categories: columns,
      reversed: true,
      labels: { style: { fontSize: "10px" } },
      title: { text: undefined },
    },
    colorAxis: {
      min: -1,
      max: 1,
      stops: [
        [0, "#f08080"], // strong negative
        [0.5, "#161616"], // neutral
        [1, "#5eead4"], // strong positive
      ],
      labels: { style: { color: "#c9c1b4" } },
    },
    legend: {
      enabled: true,
      align: "right",
      layout: "vertical",
      verticalAlign: "middle",
      symbolHeight: 180,
      itemStyle: { color: "#c9c1b4" },
    },
    tooltip: {
      formatter(this: unknown) {
        const point = (this as { point: { x: number; y: number; value: number } }).point;
        return `<b>${columns[point.y]} × ${columns[point.x]}</b><br/>r = ${point.value.toFixed(3)}`;
      },
    },
    series: [
      {
        type: "heatmap",
        name: "ρ",
        data,
        borderWidth: 0,
        dataLabels: {
          enabled: columns.length <= 10,
          color: "#0a0a0a",
          format: "{point.value:.2f}",
          style: { fontSize: "9px", textOutline: "none" },
        },
      } as Highcharts.SeriesHeatmapOptions,
    ],
  };

  return (
    <ChartFrame
      eyebrow="Correlation"
      title="Pearson correlation"
      description={`Top ${columns.length} numeric columns by absolute correlation.`}
      className={className}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </ChartFrame>
  );
}
