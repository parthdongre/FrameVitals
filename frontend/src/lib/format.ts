/**
 * Display formatters used across the report. Pure functions, no React.
 */

import { safeNum } from "./safe";

const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;

/**
 * 84_212 → "82.2 KB", 1_500_000 → "1.43 MB". Mirrors the Python helper in
 * `modules/frontend_api.py` so backend-formatted strings and frontend-formatted
 * strings line up.
 */
export function formatBytes(value: unknown): string {
  let size = Math.max(safeNum(value, 0), 0);
  let i = 0;
  while (size >= 1024 && i < BYTE_UNITS.length - 1) {
    size /= 1024;
    i += 1;
  }
  if (size >= 100) return `${size.toFixed(0)} ${BYTE_UNITS[i]}`;
  if (size >= 10) return `${size.toFixed(1)} ${BYTE_UNITS[i]}`;
  return `${size.toFixed(2)} ${BYTE_UNITS[i]}`;
}

/**
 * 0.142 → "14.2%", 0.4 → "40%". `decimals` defaults to 1.
 */
export function formatPercent(value: unknown, decimals = 1): string {
  const v = safeNum(value, 0);
  return `${(v * 100).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}%`;
}

/**
 * 128_420 → "128,420". Falls back to "—" for non-finite input.
 */
export function formatCount(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toLocaleString();
}

/**
 * 4128 → "4.13 s", 92 → "92 ms", 152_000 → "2 m 32 s".
 */
export function formatMs(value: unknown): string {
  const ms = safeNum(value, 0);
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes} m ${seconds} s`;
}

/**
 * snake_case_label → "Snake Case Label". Keeps acronyms uppercased.
 */
export function formatLabel(value: unknown): string {
  const s = typeof value === "string" ? value : String(value ?? "");
  if (!s) return "";
  return s
    .replace(/[_-]+/g, " ")
    .split(/\s+/)
    .map((word) => {
      if (word.toUpperCase() === word && word.length > 1) return word;
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

/**
 * 0.952 → "0.95". `decimals` defaults to 2.
 */
export function formatNumber(value: unknown, decimals = 2): string {
  const v = safeNum(value, NaN);
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}
