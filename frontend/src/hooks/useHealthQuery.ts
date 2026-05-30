import { useQuery } from "@tanstack/react-query";
import { getJSON } from "@/lib/api";
import type { HealthResponse } from "@/data/payload";

export const HEALTH_QUERY_KEY = ["health"] as const;

/**
 * Lightweight `/api/health` poll. Used by the top-nav status chip. Errors
 * are swallowed to a known shape so the chip can render an "offline" pip
 * without breaking layout.
 */
export function useHealthQuery() {
  return useQuery<HealthResponse>({
    queryKey: HEALTH_QUERY_KEY,
    queryFn: () => getJSON<HealthResponse>("/api/health"),
    refetchInterval: 30_000,
    refetchOnWindowFocus: false,
    retry: 1,
    staleTime: 15_000,
    // Don't bubble — the health chip should never blow up the page.
    throwOnError: false,
  });
}
