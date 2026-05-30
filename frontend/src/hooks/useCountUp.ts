import { useEffect, useRef, useState } from "react";
import { useInView } from "./useInView";

interface CountUpOptions {
  duration?: number; // seconds
  /**
   * Initial value to count from. Defaults to 0.
   */
  from?: number;
  /**
   * Custom easing — defaults to easeOutCubic.
   */
  ease?: (t: number) => number;
  /**
   * If true (default), waits until the element scrolls into view before
   * starting the tween.
   */
  whenInView?: boolean;
}

const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3);

/**
 * Count-up tween that respects reduced motion (handled at the global CSS
 * level via the `prefers-reduced-motion` block in `globals.css`, which
 * collapses transitions). Returns a ref to attach to the element and the
 * current displayed numeric value.
 */
export function useCountUp(
  to: number,
  options: CountUpOptions = {},
): [React.MutableRefObject<HTMLElement | null>, number] {
  const { duration = 0.9, from = 0, ease = easeOutCubic, whenInView = true } = options;
  const [ref, inView] = useInView<HTMLElement>({ once: true, threshold: 0.4 });
  const [value, setValue] = useState<number>(from);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (whenInView && !inView) return;
    const target = Number.isFinite(to) ? to : 0;
    const start = performance.now();
    const tick = (t: number) => {
      const elapsed = (t - start) / 1000;
      const k = Math.min(1, elapsed / Math.max(duration, 0.001));
      setValue(from + (target - from) * ease(k));
      if (k < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [inView, whenInView, to, from, duration, ease]);

  return [ref, value];
}
