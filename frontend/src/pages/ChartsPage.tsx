import { useMemo, useState } from "react";
import { Eyebrow, Hr, PageTitle, Section } from "@/components/site/SiteShell";
import { StaticChartImage } from "@/components/ui/StaticChartImage";
import { EmptyState } from "@/components/ui/EmptyState";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { motion } from "framer-motion";
import { toChartUrl } from "@/lib/chartUrl";
import { safeArr, safeStr } from "@/lib/safe";
import { formatLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ChartItem } from "@/data/payload";
import type { DashboardTelemetry } from "@/data/mockTelemetry";
import type { Route } from "@/components/site/SiteShell";

interface ChartsPageProps {
  telemetry: DashboardTelemetry | null;
  onNavigate: (r: Route) => void;
}

/**
 * Full chart gallery — every backend PNG plus the SHAP summary, with a
 * type-based filter chip strip. Uses `StaticChartImage` so each tile fades
 * in once the PNG decodes.
 */
export function ChartsPage({ telemetry, onNavigate }: ChartsPageProps) {
  const [filter, setFilter] = useState<string>("all");

  const items = useMemo<ChartItem[]>(() => {
    if (!telemetry) return [];
    const t = telemetry as unknown as Record<string, unknown>;
    const charts = safeArr<ChartItem>(t.charts);
    const shapPath = safeStr(
      (t.explainability as { summary_chart_path?: string } | undefined)
        ?.summary_chart_path,
      "",
    );
    return [
      ...charts,
      ...(shapPath
        ? [
            {
              title: "SHAP Summary",
              type: "shap_summary",
              description: "Mean |SHAP| values across the validation set.",
              path: shapPath,
            } as ChartItem,
          ]
        : []),
    ];
  }, [telemetry]);

  const types = useMemo(() => {
    const set = new Set<string>();
    items.forEach((c) => set.add(safeStr(c.type, "chart")));
    return Array.from(set).sort();
  }, [items]);

  const filtered = filter === "all" ? items : items.filter((c) => c.type === filter);

  if (!telemetry) {
    return (
      <Section className="pt-6">
        <Eyebrow>Charts</Eyebrow>
        <PageTitle subtitle="The full chart gallery — every backend PNG plus the SHAP summary plus interactive Highcharts — appears here after you analyze a dataset.">
          No analysis yet.
        </PageTitle>
        <button onClick={() => onNavigate("analyze")} className="btn-primary">
          Go to Analyze →
        </button>
      </Section>
    );
  }

  return (
    <>
      <Section className="pb-8 pt-6">
        <Eyebrow>Charts</Eyebrow>
        <PageTitle subtitle="Every chart the backend generated for the latest analysis. Static PNGs come straight from the pipeline; their interactive Highcharts twins live inside the relevant report tabs.">
          Chart gallery.
        </PageTitle>
        <p className="font-mono text-[11px] tabular-nums text-ink-3">
          {items.length} chart{items.length === 1 ? "" : "s"}
          {filter !== "all" ? ` · filtered to ${filter}` : ""}
        </p>
      </Section>

      <Hr label="Browse" />

      <Section className="space-y-8">
        {types.length > 1 ? (
          <div className="flex flex-wrap gap-2">
            <FilterChip
              active={filter === "all"}
              onClick={() => setFilter("all")}
              count={items.length}
            >
              All
            </FilterChip>
            {types.map((t) => {
              const count = items.filter((c) => c.type === t).length;
              return (
                <FilterChip
                  key={t}
                  active={filter === t}
                  onClick={() => setFilter(t)}
                  count={count}
                >
                  {formatLabel(t)}
                </FilterChip>
              );
            })}
          </div>
        ) : null}

        {filtered.length === 0 ? (
          <EmptyState
            title="No charts match the current filter"
            hint="Pick a different chart type, or clear the filter to see everything."
          />
        ) : (
          <motion.div
            variants={staggerParent}
            initial="initial"
            animate="animate"
            className="grid grid-cols-1 gap-6 sm:grid-cols-2 xl:grid-cols-3"
          >
            {filtered.map((c, i) => (
              <motion.figure
                key={`${c.path ?? i}-${c.title}`}
                variants={staggerChild}
                className="overflow-hidden rounded-md border border-line bg-bg-1 transition-colors hover:border-line-strong"
              >
                <header className="border-b border-line px-4 py-3">
                  <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">
                    {safeStr(c.type, "chart")}
                  </p>
                  <p className="mt-1 text-[14px] text-ink-1">{safeStr(c.title, "Untitled")}</p>
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
              </motion.figure>
            ))}
          </motion.div>
        )}
      </Section>
    </>
  );
}

function FilterChip({
  children,
  active,
  count,
  onClick,
}: {
  children: React.ReactNode;
  active: boolean;
  count: number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[12px] transition-colors",
        active
          ? "border-accent-line bg-accent-soft text-accent"
          : "border-line bg-bg-1 text-ink-3 hover:border-line-strong hover:text-ink-1",
      )}
    >
      <span>{children}</span>
      <span className="font-mono text-[10px] tabular-nums opacity-70">{count}</span>
    </button>
  );
}
