import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import "@/charts/theme";
import { ChartFrame } from "@/components/ui/ChartFrame";
import { EmptyState } from "@/components/ui/EmptyState";
import { safeArr, safeNum, safeStr } from "@/lib/safe";

interface FeatureImportanceBarsProps {
  importance: unknown;
  className?: string;
  /**
   * Maximum bars rendered. Default 20.
   */
  limit?: number;
  title?: string;
  description?: string;
  eyebrow?: string;
}

/**
 * Horizontal bar chart of feature importance. Accepts either the
 * `featureImportance.global_importance` or the `explainability.global_importance`
 * shape — both carry `{ feature, importance }` rows.
 */
export function FeatureImportanceBars({
  importance,
  className,
  limit = 20,
  title = "Feature importance",
  description = "Mean absolute attribution per feature on the validation set.",
  eyebrow = "Feature importance",
}: FeatureImportanceBarsProps) {
  // The payload sometimes nests the array as `.global_importance`
  // and sometimes is the array itself.
  const arr =
    safeArr<Record<string, unknown>>(
      (importance as { global_importance?: unknown })?.global_importance ?? importance,
    );

  const rows = arr
    .map((r) => ({
      feature: safeStr(r.feature ?? r.name, ""),
      importance: safeNum(r.importance ?? r.value, NaN),
    }))
    .filter((r) => r.feature && Number.isFinite(r.importance))
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
    .slice(0, limit);

  if (!rows.length) {
    return (
      <ChartFrame eyebrow={eyebrow} title={title} description={description} className={className}>
        <EmptyState
          compact
          title="No importance values"
          hint="The backend did not produce importance values for this dataset."
        />
      </ChartFrame>
    );
  }

  const options: Highcharts.Options = {
    chart: { type: "bar", height: Math.max(220, rows.length * 22 + 40) },
    title: { text: undefined },
    xAxis: {
      categories: rows.map((r) => r.feature),
      labels: { style: { fontSize: "11px" } },
    },
    yAxis: {
      title: { text: "importance" },
    },
    legend: { enabled: false },
    plotOptions: {
      bar: {
        borderWidth: 0,
        pointPadding: 0.05,
        groupPadding: 0,
      },
    },
    tooltip: {
      pointFormat: "<b>{point.y:.4f}</b>",
    },
    series: [
      {
        type: "bar",
        name: "importance",
        data: rows.map((r) => Number(r.importance.toFixed(6))),
        color: "var(--accent)",
      } as Highcharts.SeriesBarOptions,
    ],
  };

  return (
    <ChartFrame eyebrow={eyebrow} title={title} description={`Top ${rows.length} feature${rows.length === 1 ? "" : "s"}.`} className={className}>
      <HighchartsReact highcharts={Highcharts} options={options} />
    </ChartFrame>
  );
}
