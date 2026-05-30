import { useMutation } from "@tanstack/react-query";
import { ApiError, postFormData } from "@/lib/api";
import type { DashboardTelemetry, AnalysisMode } from "@/data/mockTelemetry";

export interface AnalyzeDatasetInput {
  file: File;
  analysisMode: AnalysisMode;
  targetColumn?: string | null;
}

/**
 * Wraps `POST /api/analyze`. Returns the same `DashboardTelemetry` shape the
 * legacy code already consumes, so existing pages keep compiling. The error
 * path is now unified through `ApiError` so the new analyze page can render
 * a structured error card with the server's `error` field.
 */
export function useAnalyzeDatasetMutation() {
  return useMutation<DashboardTelemetry, Error, AnalyzeDatasetInput>({
    mutationFn: async ({ file, analysisMode, targetColumn }) => {
      const formData = new FormData();
      formData.append("dataset", file);
      formData.append("analysis_mode", analysisMode);
      if (targetColumn) formData.append("target_column", targetColumn);

      try {
        return await postFormData<DashboardTelemetry>("/api/analyze", formData, {
          // Deep / research mode on a multi-thousand-row dataset can take
          // 30-90 seconds; give analyze five minutes before the abort fires.
          timeoutMs: 5 * 60 * 1000,
        });
      } catch (err) {
        if (err instanceof ApiError) {
          throw new Error(err.message || `Dataset analysis failed (${err.status}).`);
        }
        throw err instanceof Error ? err : new Error("Dataset analysis failed.");
      }
    },
  });
}
