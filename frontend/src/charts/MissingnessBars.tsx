import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import "@/charts/theme";
import { ChartFrame } from "@/components/ui/ChartFrame";
import { EmptyState } from "@/components/ui/EmptyState";
import { safeNum, safeObj } from "@/lib/safe";

interface MissingnessBarsProps {
  /**
   * Either `profile.missing_percent` (column → percent) or
   * `profile.missing_counts` (column → count). When both are passed via
   * `percent` and `total`, the chart prefers `percent`.
   */
  percent?: unknown;
  /**
   * Optional total row count, used to compute percent from raw counts when
   * `percent` is missing.
   */
  total?: number;
  className?: string;
  /**
   * Maximum bars to show. Default 16 (sorted descending by missingness).
   */
  limit?: number;
}

/**
 * Horizontal bar chart of per-column missingness percentages. Always sorted
 * descending so the worst offenders are at the top.
 */
export function MissingnessBars({
  percent,
  total = 0,
  className,
  limit = 16,
}: MissingnessBarsProps) {
  const pctMap = safeObj(percent, {} as Record<string, unknown>);
  const totalRows = safeNum(total, 0);

  const rows = Object.entries(pctMap)
    .map(([col, raw]) => {
      const v = safeNum(raw, 0);
      // Heuristic: if values look like raw counts (>1) and we know the total,
      // convert to percent. Otherwise treat as percent already.
      const pct = totalRows > 0 && v > 1 ? (v / totalRows) * 100 : v;
      return [col, pct] as const;
    })
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit);

  if (rows.length === 0) {
    return (
      <ChartFrame
        eyebrow="Missingness"
        title="Per-column missing rate"
        description="Columns ranked by their share of missing values."
        className={className}
      >
        <EmptyState
          compact
          title="No missing values"
          hint="Every column in this dataset is fully populated."
        />
      </ChartFrame>
    );
  }

  const categories = rows.map(([col]) => col);
  const data = rows.map(([, v]) => Number(v.toFixed(2)));

  const options: Highcharts.Options = {
    chart: { type: "bar", height: Math.max(180, rows.length * 22 + 40) },
    title: { text: undefined },
    xAxis: { categories, labels: { style: { fontSize: "11px" } } },
    yAxis: {
      min: 0,
      max: Math.min(100, Math.max(...data) * 1.15 + 1),
      title: { text: "% missing" },
      labels: { format: "{value}%" },
    },
    tooltip: {
      pointFormat: "<b>{point.y:.2f}%</b> missing",
    },
    legend: { enabled: false },
    plotOptions: {
      bar: {
        borderWidth: 0,
        pointPadding: 0.04,
        groupPadding: 0,
      },
    },
    series: [
      {
        type: "bar",
        name: "Missing",
        data,
        color: "var(--accent)",
      } as Highcharts.SeriesBarOptions,
    ],
  };

  return (
    <ChartFrame
      eyebrow="Missingness"
      title="Per-column missing rate"
      description={`${rows.length} column${rows.length === 1 ? "" : "s"} with missing values, sorted by share.`}
      className={className}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </ChartFrame>
  );
}
