import { useCountUp } from "@/hooks/useCountUp";
import { cn } from "@/lib/utils";

interface AnimatedNumberProps {
  value: number;
  /**
   * Tween duration in seconds. Default 0.9s.
   */
  duration?: number;
  /**
   * Initial value to count from. Default 0.
   */
  from?: number;
  /**
   * Number of decimals to render. Default 0 (rounded integer).
   */
  decimals?: number;
  /**
   * Inserts thousands separators. Default true.
   */
  group?: boolean;
  prefix?: string;
  suffix?: string;
  className?: string;
}

/**
 * Drop-in count-up display for KPI cards. Renders tabular-nums and animates
 * the underlying number once the element scrolls into view.
 *
 * Usage:
 *   <AnimatedNumber value={128_420} />
 *   <AnimatedNumber value={99.2} decimals={1} suffix="%" />
 */
export function AnimatedNumber({
  value,
  duration = 0.9,
  from = 0,
  decimals = 0,
  group = true,
  prefix,
  suffix,
  className,
}: AnimatedNumberProps) {
  const [ref, current] = useCountUp(value, { duration, from });
  const safe = Number.isFinite(current) ? current : 0;
  const formatted = group
    ? safe.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })
    : safe.toFixed(decimals);

  return (
    <span
      ref={ref as React.RefObject<HTMLSpanElement>}
      className={cn("tnum", className)}
    >
      {prefix}
      {formatted}
      {suffix}
    </span>
  );
}
