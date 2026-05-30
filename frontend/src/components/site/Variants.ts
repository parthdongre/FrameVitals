import type { Variants } from "framer-motion";

/**
 * Editorial motion language for the v3 rebuild.
 *
 * - Page transitions: 350ms fade + slight Y, easeOut.
 * - Tab swap: 250ms crossfade.
 * - Stagger reveals: 40ms apart, easeOut.
 * - Hover: subtle border-color + 1px lift.
 * - Button press: tap-down compress + accent border-glow.
 *
 * Every variant respects `prefers-reduced-motion` via the
 * `useReducedMotion()` hook on the consumer side. When reduced motion is on,
 * pass `false` for the `motion` argument and the helper builders below will
 * collapse the durations to ~0.
 */

export const ease = [0.22, 1, 0.36, 1] as const;

export const pageVariants: Variants = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease } },
  exit: { opacity: 0, y: -6, transition: { duration: 0.2, ease } },
};

export const tabVariants: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.25, ease } },
  exit: { opacity: 0, transition: { duration: 0.18, ease } },
};

export const staggerParent: Variants = {
  initial: {},
  animate: { transition: { staggerChildren: 0.04, delayChildren: 0.05 } },
};

export const staggerChild: Variants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3, ease } },
};

export const hoverLift = {
  whileHover: {
    y: -1,
    borderColor: "var(--line-strong)",
    transition: { duration: 0.18, ease },
  },
  whileTap: { scale: 0.992, transition: { duration: 0.1, ease } },
} as const;

export const buttonPress = {
  whileHover: {
    boxShadow: "inset 0 0 0 1px var(--accent-line)",
    transition: { duration: 0.18, ease },
  },
  whileTap: { scale: 0.97, transition: { duration: 0.08, ease } },
} as const;

/**
 * Helper for callers that want a no-op variant set when reduced motion is on.
 * Pass the result of `useReducedMotion()` to the second argument.
 */
export function withMotion<V extends Variants>(variants: V, reduced: boolean): V {
  if (!reduced) return variants;
  const stripped = {} as Variants;
  for (const key of Object.keys(variants)) {
    stripped[key] = { transition: { duration: 0 } };
  }
  return stripped as V;
}
