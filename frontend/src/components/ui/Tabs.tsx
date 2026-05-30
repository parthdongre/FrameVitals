import type { ComponentType, ReactNode } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export interface TabItem {
  id: string;
  label: ReactNode;
  /**
   * Optional Lucide icon to render to the left of the label.
   */
  Icon?: ComponentType<{ className?: string; size?: number }>;
  /**
   * If false, render with reduced opacity to signal "no data". The tab is
   * still clickable so the user can confirm the empty-state message.
   */
  hasData?: boolean;
  /**
   * Optional badge shown inline (e.g. counts).
   */
  badge?: ReactNode;
}

interface TabsProps {
  items: TabItem[];
  activeId: string;
  onChange: (id: string) => void;
  /**
   * shared `layoutId` for the active underline so framer animates the move
   * smoothly. Override only if you nest multiple `<Tabs>` on one screen.
   */
  layoutId?: string;
  className?: string;
}

/**
 * Editorial pill-tab strip. The active underline animates between tabs via
 * framer's shared layout. Used for the 17 report sub-tabs and any future
 * intra-page navigation.
 *
 * The strip wraps to multiple lines on narrow viewports rather than scrolling
 * so users can see every available tab at a glance.
 */
export function Tabs({ items, activeId, onChange, layoutId = "tab-underline", className }: TabsProps) {
  return (
    <div
      role="tablist"
      aria-orientation="horizontal"
      className={cn(
        "flex flex-wrap items-center gap-x-1 gap-y-2 border-b border-line",
        className,
      )}
    >
      {items.map((item) => {
        const active = item.id === activeId;
        const dim = item.hasData === false;
        const Icon = item.Icon;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            aria-controls={`tabpanel-${item.id}`}
            id={`tab-${item.id}`}
            onClick={() => onChange(item.id)}
            className={cn(
              "relative inline-flex items-center gap-2 px-3 py-2 text-[13px] transition-colors duration-150",
              active
                ? "text-ink-1"
                : "text-ink-3 hover:text-ink-1",
              dim && !active && "opacity-60",
            )}
          >
            {Icon ? <Icon className="h-3.5 w-3.5" /> : null}
            <span>{item.label}</span>
            {item.badge ? (
              <span className="ml-1 inline-flex items-center justify-center rounded-full border border-line bg-bg-2 px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-ink-3">
                {item.badge}
              </span>
            ) : null}
            {active ? (
              <motion.span
                layoutId={layoutId}
                className="absolute inset-x-2 -bottom-px h-px bg-accent"
                transition={{ type: "spring", stiffness: 500, damping: 32 }}
              />
            ) : null}
          </button>
        );
      })}
    </div>
  );
}
