/**
 * Safe accessors for the analyze payload.
 *
 * Every panel reads through these so a missing or malformed field can never
 * white-screen a tab. The previous rebuild attempt died exactly here — a
 * single `undefined` slipped through and took down the page. Now every
 * destructure is preceded by a `safeArr` / `safeObj` and every numeric
 * comparison is preceded by a `safeNum`.
 */

export const safeNum = (v: unknown, fallback = 0): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;

export const safeStr = (v: unknown, fallback = ""): string =>
  typeof v === "string" ? v : fallback;

export const safeBool = (v: unknown, fallback = false): boolean =>
  typeof v === "boolean" ? v : fallback;

export const safeArr = <T = unknown>(v: unknown, fallback: T[] = []): T[] =>
  Array.isArray(v) ? (v as T[]) : fallback;

export const safeObj = <T extends object>(v: unknown, fallback: T): T =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as T) : fallback;

/**
 * `isPresent` returns true when the value carries any content. Empty objects
 * and empty arrays are treated as absent so panels can use a single predicate
 * for their `hasData` checks.
 */
export const isPresent = (v: unknown): boolean => {
  if (v === undefined || v === null) return false;
  if (typeof v === "string") return v.length > 0;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "object") return Object.keys(v as object).length > 0;
  if (typeof v === "number") return Number.isFinite(v);
  return Boolean(v);
};

/**
 * Pulls a numeric value from an unknown record by key, returning `fallback`
 * if missing / non-finite. Convenience wrapper for KPI rendering.
 */
export const pickNum = (
  obj: unknown,
  key: string,
  fallback = 0,
): number => safeNum((safeObj(obj, {} as Record<string, unknown>) as Record<string, unknown>)[key], fallback);

/**
 * Pulls a string value from an unknown record by key.
 */
export const pickStr = (
  obj: unknown,
  key: string,
  fallback = "",
): string => safeStr((safeObj(obj, {} as Record<string, unknown>) as Record<string, unknown>)[key], fallback);
