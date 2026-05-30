import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface ChartFrameProps {
  /**
   * Chart title rendered above the canvas.
   */
  title?: ReactNode;
  /**
   * Optional eyebrow (mono uppercase) for context (e.g. "DECOMPOSITION").
   */
  eyebrow?: string;
  /**
   * Subtitle / description below the title.
   */
  description?: ReactNode;
  /**
   * Slot for a small affordance shown in the top-right (e.g. "Open as PNG").
   */
  action?: ReactNode;
  /**
   * The chart itself (Highcharts, custom canvas, etc.).
   */
  children: ReactNode;
  /**
   * Optional height for the chart canvas. The frame has a fixed height to
   * prevent layout flashes on first paint.
   */
  height?: number | string;
  className?: string;
}

/**
 * Hairline card wrapper used by every interactive Highcharts component. It
 * keeps title/description styling consistent and gives the chart a fixed
 * canvas height so initialization doesn't push surrounding content.
 */
export function ChartFrame({
  title,
  eyebrow,
  description,
  action,
  children,
  height,
  className,
}: ChartFrameProps) {
  return (
    <figure
      className={cn(
        "relative rounded-md border border-line bg-bg-1 transition-colors hover:border-line-strong",
        className,
      )}
    >
      {(title || eyebrow || description || action) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-4 py-3">
          <div>
            {eyebrow ? (
              <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">
                {eyebrow}
              </p>
            ) : null}
            {title ? (
              <p className="mt-1 text-[14px] text-ink-1">{title}</p>
            ) : null}
            {description ? (
              <p className="mt-1 text-[12px] leading-5 text-ink-3">{description}</p>
            ) : null}
          </div>
          {action ? <div className="shrink-0">{action}</div> : null}
        </header>
      )}
      <div
        className="px-2 py-3 sm:px-4 sm:py-4"
        style={height ? { height } : undefined}
      >
        {children}
      </div>
    </figure>
  );
}
