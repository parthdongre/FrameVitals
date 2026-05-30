import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface KeyValueItem {
  label: ReactNode;
  value: ReactNode;
  /**
   * Optional eyebrow (typically a unit or category).
   */
  eyebrow?: string;
  /**
   * If true, the value renders mono and tabular-nums (KPI feel).
   */
  mono?: boolean;
  /**
   * Force this row to span 2 columns at the current breakpoint.
   */
  wide?: boolean;
}

interface KeyValueGridProps {
  items: KeyValueItem[];
  /**
   * Number of columns at the >=md breakpoint. Default 2. Set to 3 or 4 for
   * dense KPI strips.
   */
  cols?: 1 | 2 | 3 | 4;
  className?: string;
  /**
   * Render compact (smaller padding, tighter rhythm). Default false.
   */
  compact?: boolean;
}

/**
 * Generic key/value display used across diagnostics, profile, ML lab. Renders
 * each entry as an editorial label/value pair separated by hairlines.
 */
export function KeyValueGrid({ items, cols = 2, className, compact }: KeyValueGridProps) {
  if (!items || items.length === 0) return null;
  const colClass =
    cols === 1
      ? "grid-cols-1"
      : cols === 2
        ? "grid-cols-1 md:grid-cols-2"
        : cols === 3
          ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
          : "grid-cols-2 lg:grid-cols-4";
  return (
    <dl className={cn("grid border-t border-l border-line", colClass, className)}>
      {items.map((item, i) => (
        <div
          key={`${i}-${typeof item.label === "string" ? item.label : i}`}
          className={cn(
            "border-b border-r border-line",
            compact ? "px-3 py-2" : "px-4 py-3",
            item.wide && cols >= 2 ? "md:col-span-2" : "",
          )}
        >
          {item.eyebrow ? (
            <div className="mb-1 font-mono text-[10px] uppercase tracking-[0.28em] text-ink-3">
              {item.eyebrow}
            </div>
          ) : null}
          <dt className="text-[12px] uppercase tracking-wider text-ink-3">{item.label}</dt>
          <dd
            className={cn(
              "mt-1 text-[14px] text-ink-1",
              item.mono && "font-mono tabular-nums text-[13px]",
            )}
          >
            {item.value ?? <span className="text-ink-4">—</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
