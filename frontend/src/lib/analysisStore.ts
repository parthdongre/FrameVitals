import type { DashboardTelemetry } from "@/data/mockTelemetry";

/**
 * Module-level cache for the latest analyze result.
 *
 * The hash router has no React Context — pages reach for the analysis
 * directly. Once @tanstack/react-query is wired into the analyze mutation we
 * could promote this to its own query key, but a single-slot module cache is
 * cheaper, faster, and matches the design's "single payload feeds the whole
 * report" model.
 *
 * Subscribers are notified whenever the cache changes, so headless callers
 * (e.g. an Ask Anything panel that wants to react to a re-analyze) can
 * stay in sync without prop drilling.
 */

type Listener = (value: DashboardTelemetry | null) => void;

let cached: DashboardTelemetry | null = null;
const listeners = new Set<Listener>();

export function setAnalysis(value: DashboardTelemetry | null): void {
  cached = value;
  for (const fn of listeners) {
    try {
      fn(cached);
    } catch {
      /* swallow listener errors */
    }
  }
}

export function getAnalysis(): DashboardTelemetry | null {
  return cached;
}

export function clearAnalysis(): void {
  setAnalysis(null);
}

export function subscribeAnalysis(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
