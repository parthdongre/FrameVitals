import HighchartsReact from "highcharts-react-official";
import Highcharts from "highcharts";
import "@/charts/theme";
import { ChartFrame } from "@/components/ui/ChartFrame";
import { EmptyState } from "@/components/ui/EmptyState";
import { safeArr, safeNum, safeStr } from "@/lib/safe";

interface LeaderboardBarsProps {
  /**
   * Either the full `modelLeaderboard` object or just the `leaderboard[]` array.
   */
  leaderboard?: unknown;
  /**
   * Optional explicit primary metric label (overrides whatever the backend
   * reports).
   */
  primaryMetric?: string;
  className?: string;
}

/**
 * Horizontal bar chart of cross-validated primary metric per model. The
 * winner is highlighted in the accent color; the rest fall back to muted
 * cream so the comparison stays calm.
 */
export function LeaderboardBars({ leaderboard, primaryMetric, className }: LeaderboardBarsProps) {
  // Accept either { leaderboard: [...] } or the array directly.
  const root = (leaderboard ?? {}) as {
    leaderboard?: unknown[];
    winner?: { model?: string };
    primary_metric?: string;
  };
  const rows = safeArr<Record<string, unknown>>(
    Array.isArray(leaderboard) ? leaderboard : root.leaderboard,
  );
  const metricName = safeStr(primaryMetric, safeStr(root.primary_metric, "score"));
  const winnerName = safeStr(root.winner?.model, "");

  const series = rows
    .map((r) => {
      const model = safeStr(r.model, "");
      const score =
        safeNum(r.primary_score, safeNum(r.primary_metric_value, NaN));
      return { model, score };
    })
    .filter((r) => r.model && Number.isFinite(r.score))
    .sort((a, b) => b.score - a.score);

  if (!series.length) {
    return (
      <ChartFrame
        eyebrow="ML Lab"
        title="Model leaderboard"
        description="Cross-validated primary metric per model."
        className={className}
      >
        <EmptyState
          compact
          title="No leaderboard rows"
          hint="Run with a target column in standard or deeper mode to populate the leaderboard."
        />
      </ChartFrame>
    );
  }

  const data = series.map((r) => ({
    name: r.model,
    y: Number(r.score.toFixed(4)),
    color: r.model === winnerName ? "#5eead4" : "#c9c1b4",
  }));

  const options: Highcharts.Options = {
    chart: { type: "bar", height: Math.max(200, series.length * 30 + 40) },
    title: { text: undefined },
    xAxis: {
      categories: series.map((r) => r.model),
      labels: { style: { fontSize: "11px" } },
    },
    yAxis: {
      title: { text: metricName },
      labels: { format: "{value:.3f}" },
    },
    legend: { enabled: false },
    tooltip: {
      pointFormat: `<b>{point.y:.4f}</b> · ${metricName}`,
    },
    plotOptions: {
      bar: {
        borderWidth: 0,
        pointPadding: 0.05,
        groupPadding: 0,
        dataLabels: {
          enabled: true,
          format: "{point.y:.3f}",
          style: { color: "#c9c1b4", textOutline: "none", fontSize: "10px" },
        },
      },
    },
    series: [
      {
        type: "bar",
        name: metricName,
        data,
      } as Highcharts.SeriesBarOptions,
    ],
  };

  return (
    <ChartFrame
      eyebrow="ML Lab"
      title="Model leaderboard"
      description={`${series.length} model${series.length === 1 ? "" : "s"} · primary metric ${metricName}${
        winnerName ? ` · winner ${winnerName}` : ""
      }`}
      className={className}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </ChartFrame>
  );
}
