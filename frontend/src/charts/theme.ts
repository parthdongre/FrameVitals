import Highcharts from "highcharts";

/**
 * Editorial Highcharts theme.
 *
 * Imported once for side-effect from the entry point of any chart module.
 * Idempotent: calling it twice does no harm. Sets the cream + near-black +
 * teal palette across every chart in the app.
 */

let applied = false;

export function applyHighchartsTheme(): void {
  if (applied) return;
  applied = true;

  Highcharts.setOptions({
    chart: {
      backgroundColor: "transparent",
      style: { fontFamily: "Inter, system-ui, sans-serif" },
      animation: { duration: 350 },
      spacing: [8, 8, 8, 8],
    },
    colors: [
      "#5eead4", // accent
      "#c9c1b4", // ink-2
      "#f5efe6", // ink-1
      "#8a8478", // ink-3
      "#5a5650", // ink-4
      "#84d8b8", // ok
      "#f5b14a", // warn
      "#f08080", // bad
    ],
    title: {
      style: { color: "#f5efe6", fontWeight: "800" },
    },
    subtitle: {
      style: { color: "#c9c1b4", fontWeight: "600" },
    },
    xAxis: {
      lineColor: "rgba(255,255,255,0.16)",
      tickColor: "rgba(255,255,255,0.16)",
      gridLineColor: "rgba(255,255,255,0.04)",
      labels: { style: { color: "#c9c1b4", fontSize: "11px", fontWeight: "600" } },
      title: { style: { color: "#8a8478", fontWeight: "700" } },
    },
    yAxis: {
      lineColor: "rgba(255,255,255,0.16)",
      tickColor: "rgba(255,255,255,0.16)",
      gridLineColor: "rgba(255,255,255,0.06)",
      labels: { style: { color: "#c9c1b4", fontSize: "11px", fontWeight: "600" } },
      title: { style: { color: "#8a8478", fontWeight: "700" } },
    },
    legend: {
      itemStyle: { color: "#c9c1b4", fontWeight: "600" },
      itemHoverStyle: { color: "#f5efe6" },
      itemHiddenStyle: { color: "#5a5650" },
    },
    tooltip: {
      backgroundColor: "rgba(17,17,17,0.95)",
      borderColor: "rgba(255,255,255,0.16)",
      borderRadius: 4,
      style: { color: "#f5efe6", fontSize: "12px", fontWeight: "600" },
    },
    plotOptions: {
      series: {
        animation: { duration: 400 },
        dataLabels: { color: "#c9c1b4", style: { fontWeight: "700" } },
        marker: { lineColor: "rgba(255,255,255,0.16)" },
      },
    },
    credits: { enabled: false },
    accessibility: { enabled: false },
  });
}

// Apply on import — every chart module starts with `import "@/charts/theme"`.
applyHighchartsTheme();
