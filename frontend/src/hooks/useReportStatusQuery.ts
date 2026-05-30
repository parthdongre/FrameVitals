import { useQuery } from "@tanstack/react-query";
import { getJSON } from "@/lib/api";
import type { ReportStatusResponse } from "@/data/payload";

export const reportStatusKey = (datasetId: string | null | undefined) =>
  ["reportStatus", datasetId] as const;

/**
 * Polls `/api/report-status/<id>` until the PDF is ready. Stops polling on
 * `ready` or `failed` to avoid hammering Flask. Used by the "Download report"
 * button on the Overview / Cleaning tabs.
 */
export function useReportStatusQuery(datasetId: string | null | undefined, enabled = true) {
  return useQuery<ReportStatusResponse>({
    queryKey: reportStatusKey(datasetId),
    enabled: enabled && Boolean(datasetId),
    queryFn: () => getJSON<ReportStatusResponse>(`/api/report-status/${datasetId}`),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      if (status === "ready" || status === "failed" || status === "missing") return false;
      return 1500;
    },
    refetchOnWindowFocus: false,
    throwOnError: false,
    retry: 0,
  });
}
