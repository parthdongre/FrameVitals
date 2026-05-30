import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { StaticChartImage } from "@/components/ui/StaticChartImage";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { safeArr, safeStr } from "@/lib/safe";
import { toChartUrl } from "@/lib/chartUrl";
import { useHashRoute } from "@/router/HashRouter";
import type { ChartItem } from "@/data/payload";
import type { TabComponentProps } from "./tabRegistry";

/**
 * In-tab Charts shortcut. Shows a tight grid of every backend PNG plus the
 * SHAP summary (if any), and links to the full /charts gallery for a
 * larger-format view.
 */
export default function ChartsTab({ analysis }: TabComponentProps) {
  const { navigate } = useHashRoute();
  const t = analysis as unknown as Record<string, unknown>;
  const charts = safeArr<ChartItem>(t.charts);
  const shapPath = safeStr(
    (t.explainability as { summary_chart_path?: string } | undefined)?.summary_chart_path,
    "",
  );

  const items: ChartItem[] = [
    ...charts,
    ...(shapPath
      ? [
          {
            title: "SHAP Summary",
            type: "shap_summary",
            description: "Mean |SHAP| values across the validation set.",
            path: shapPath,
          },
        ]
      : []),
  ];

  if (!items.length) {
    return (
      <EmptyState
        title="No charts produced for this dataset"
        hint="The chart planner skipped this run. Try analyzing a richer dataset to generate the gallery."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-6">
      <motion.section variants={staggerChild} className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <Eyebrow className="mb-2">Generated charts</Eyebrow>
          <p className="text-[13px] leading-6 text-ink-3">
            {items.length} static chart{items.length === 1 ? "" : "s"}.
          </p>
        </div>
        <button
          onClick={() => navigate("charts")}
          className="btn-ghost"
          type="button"
        >
          Open full gallery →
        </button>
      </motion.section>

      <motion.div
        variants={staggerChild}
        className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3"
      >
        {items.map((c, i) => (
          <figure
            key={`${c.path ?? i}`}
            className="overflow-hidden rounded-md border border-line bg-bg-1 transition-colors hover:border-line-strong"
          >
            <header className="border-b border-line px-4 py-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">
                {safeStr(c.type, "chart")}
              </p>
              <p className="mt-1 text-[13px] text-ink-1">{safeStr(c.title, "Untitled")}</p>
            </header>
            <StaticChartImage
              src={toChartUrl(c.path)}
              alt={safeStr(c.title, "chart")}
              className="rounded-none"
            />
            {c.description ? (
              <figcaption className="border-t border-line px-4 py-3 text-[12px] leading-5 text-ink-3">
                {safeStr(c.description, "")}
              </figcaption>
            ) : null}
          </figure>
        ))}
      </motion.div>
    </motion.div>
  );
}
