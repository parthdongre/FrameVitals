import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface EmptyStateProps {
  /**
   * Headline. Defaults to the design's canonical "Not available for this dataset".
   */
  title?: string;
  /**
   * Secondary copy. Use it to hint at why the panel is empty (e.g. "Run with a
   * target column to populate ML Lab").
   */
  hint?: ReactNode;
  /**
   * Optional small icon glyph rendered above the title. Pass a Lucide icon
   * component or any ReactNode.
   */
  icon?: ReactNode;
  className?: string;
  /**
   * Compact variant trims padding for inline use inside cards.
   */
  compact?: boolean;
}

/**
 * The single canonical "no data" surface. Every report tab renders this when
 * its `hasData` predicate fails. Treating empty payload sections as a normal
 * render path is what keeps a single missing field from white-screening the
 * whole page.
 */
export function EmptyState({
  title = "Not available for this dataset",
  hint,
  icon,
  className,
  compact,
}: EmptyStateProps) {
  return (
    <div
      role="status"
      className={cn(
        "rounded-md border border-line bg-bg-1 text-center",
        compact ? "p-6" : "p-12",
        className,
      )}
    >
      {icon ? (
        <div className="mx-auto mb-4 inline-flex h-10 w-10 items-center justify-center rounded-full border border-line text-ink-3">
          {icon}
        </div>
      ) : null}
      <p
        className={cn(
          "font-display text-ink-1 text-balance",
          compact ? "text-[18px] leading-7" : "text-[22px] leading-8 sm:text-[26px] sm:leading-9",
        )}
      >
        {title}
      </p>
      {hint ? (
        <p className="mt-2 text-[13px] leading-6 text-ink-3 text-pretty">{hint}</p>
      ) : null}
    </div>
  );
}
