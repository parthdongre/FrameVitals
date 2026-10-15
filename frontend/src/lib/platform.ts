export interface SourceInfo {
  kind: string;
  analyses: string[];
  modes: string[];
  supports_focus: boolean;
  supports_reference: boolean;
  limitations: string[];
}
export interface SourcePreview {
  filename: string;
  size_bytes: number;
  source: SourceInfo;
  preview: {
    columns?: string[];
    rows?: unknown[][];
    shape?: number[];
    dtype?: string;
  } | null;
}
export interface PlatformResult {
  schema_version: string;
  id: string;
  filename: string;
  status: string;
  source: SourceInfo;
  elapsed_ms: number | null;
  score_interpretation: string;
  execution: Record<string, unknown>;
  diagnostics: {
    name: string;
    status: string;
    reason?: string;
    method?: string;
    accuracy: string;
    coverage: string;
  }[];
  analysis: Record<string, unknown>;
  findings: {
    id?: string;
    title?: string;
    name?: string;
    summary?: string;
    severity?: string;
    recommendation?: string;
  }[];
  change: Record<string, unknown> | null;
}
let latest: PlatformResult | null = null;
export function getPlatformResult() {
  return latest;
}
export function setPlatformResult(value: PlatformResult | null) {
  latest = value;
}
export const asObject = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
export function display(v: unknown): string {
  if (v === null || v === undefined) return "Unavailable";
  if (typeof v === "number")
    return Number.isFinite(v)
      ? v.toLocaleString(undefined, { maximumFractionDigits: 4 })
      : "Unavailable";
  if (typeof v === "boolean") return v ? "Yes" : "No";
  if (Array.isArray(v))
    return v.length < 10 && v.every((x) => typeof x !== "object")
      ? v.join(" × ")
      : `${v.length} items`;
  if (typeof v === "object") return `${Object.keys(v).length} fields`;
  return String(v);
}
export const label = (key: string) =>
  key.replace(/_/g, " ").replace(/\./g, " · ");
