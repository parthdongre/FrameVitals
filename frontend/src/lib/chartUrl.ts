/**
 * Chart URL helper.
 *
 * The backend payload's `chart.path` field can come in several shapes:
 *
 *   - Bare filename:               "abc123_correlation_heatmap.png"
 *   - Repo-relative w/ "charts/":  "charts/abc123_dist_age.png"
 *   - Repo-relative w/ "static":   "static/charts/abc123_dist_age.png"
 *   - Absolute path:               "/static/charts/abc123_health_components.png"
 *
 * `toChartUrl` normalizes all four to `/static/charts/<filename>`, which is
 * the canonical mount served by Flask (and the Vite dev proxy in development).
 *
 * The backend currently writes paths in the second form ("charts/<file>.png")
 * because it stores them relative to the repo's static root. Always reduce
 * to the bare filename and rebuild from there to dodge this class of bug.
 */
export function toChartUrl(path: string | null | undefined): string {
  if (!path) return "";
  const trimmed = String(path).trim();
  if (!trimmed) return "";
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  // Bare filename or any of the relative variants — pull off the basename
  // and re-anchor at the canonical mount.
  const filename = trimmed.split(/[\\/]/).pop() ?? trimmed;
  if (!filename) return "";
  return `/static/charts/${filename}`;
}
