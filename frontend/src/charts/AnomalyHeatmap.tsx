import { useMemo } from "react";
import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import HeatmapModule from "highcharts/modules/heatmap";
import "@/charts/theme";
import { ChartFrame } from "@/components/ui/ChartFrame";
import { EmptyState } from "@/components/ui/EmptyState";
import type { AnomaliesV2 } from "@/data/payload";
import { safeArr, safeNum } from "@/lib/safe";

if (typeof (HeatmapModule as unknown as (h: typeof Highcharts) => void) === "function") {
  try {
    (HeatmapModule as unknown as (h: typeof Highcharts) => void)(Highcharts);
  } catch {
    /* already registered */
  }
}

interface AnomalyHeatmapProps {
  anomalies: AnomaliesV2 | undefined | null;
  className?: string;
  /**
   * Maximum rows to show (top-N by ensemble score).
   */
  limit?: number;
}

/**
 * Per-row × per-detector heatmap of anomaly scores. Builds its data straight
 * from `anomaliesV2.top_rows`, where each row carries a `row_index`,
 * `ensemble`, and one numeric column per detector. Cells are colored on a
 * 0-1 teal-to-rose scale so high-agreement rows pop visually.
 */
export function AnomalyHeatmap({ anomalies, className, limit = 16 }: AnomalyHeatmapProps) {
  const detectors = safeArr<string>(anomalies?.detectors_run);
  const topRows = safeArr<Record<string, unknown>>(anomalies?.top_rows).slice(0, limit);

  const data = useMemo(() => {
    const points: Array<[number, number, number]> = [];
    topRows.forEach((row, y) => {
      detectors.forEach((det, x) => {
        const v = safeNum(row[det], NaN);
        if (Number.isFinite(v)) {
          points.push([x, y, Number(v.toFixed(3))]);
        }
      });
    });
    return points;
  }, [topRows, detectors]);

  if (!detectors.length || !topRows.length) {
    return (
      <ChartFrame
        eyebrow="Anomalies"
        title="Per-row detector heatmap"
        description="Score each detector assigned to the most flagged rows."
        className={className}
      >
        <EmptyState
          compact
          title="No flagged rows to chart"
          hint="Anomaly ensemble produced no rows above the threshold for this dataset."
        />
      </ChartFrame>
    );
  }

  const yCategories = topRows.map((row, idx) => {
    const i = row.row_index;
    return typeof i === "number" || typeof i === "string" ? `row ${i}` : `#${idx + 1}`;
  });

  const options: Highcharts.Options = {
    chart: {
      type: "heatmap",
      height: Math.max(220, topRows.length * 22 + 60),
    },
    title: { text: undefined },
    xAxis: {
      categories: detectors,
      labels: { rotation: -25, style: { fontSize: "10px" } },
    },
    yAxis: {
      categories: yCategories,
      reversed: true,
      title: { text: undefined },
      labels: { style: { fontSize: "10px" } },
    },
    colorAxis: {
      min: 0,
      max: 1,
      stops: [
        [0, "#161616"],
        [0.5, "#5eead4"],
        [1, "#f08080"],
      ],
      labels: { style: { color: "#c9c1b4" } },
    },
    legend: {
      enabled: true,
      align: "right",
      layout: "vertical",
      verticalAlign: "middle",
      symbolHeight: 160,
    },
    tooltip: {
      formatter(this: unknown) {
        const point = (this as { point: { x: number; y: number; value: number } }).point;
        return `<b>${yCategories[point.y]} · ${detectors[point.x]}</b><br/>score = ${point.value.toFixed(3)}`;
      },
    },
    series: [
      {
        type: "heatmap",
        name: "score",
        data,
        borderWidth: 0,
        dataLabels: {
          enabled: detectors.length <= 8,
          color: "#0a0a0a",
          format: "{point.value:.2f}",
          style: { fontSize: "9px", textOutline: "none" },
        },
      } as Highcharts.SeriesHeatmapOptions,
    ],
  };

  return (
    <ChartFrame
      eyebrow="Anomalies"
      title="Per-row detector heatmap"
      description={`Top ${topRows.length} flagged row${topRows.length === 1 ? "" : "s"} × ${detectors.length} detector${detectors.length === 1 ? "" : "s"}.`}
      className={className}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </ChartFrame>
  );
}
