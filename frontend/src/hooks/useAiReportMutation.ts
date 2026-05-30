import { useMutation } from "@tanstack/react-query";
import { postJSON } from "@/lib/api";

export interface AiReportResponse {
  source: string;
  text: string;
  deferred?: boolean;
}

/**
 * Triggers on-demand AI report generation. The pipeline skips the LLM call
 * during /api/analyze by default (saves 5-30 seconds), so the AI Report tab
 * gets an explicit "Generate" button that calls this hook.
 */
export function useAiReportMutation() {
  return useMutation<AiReportResponse, Error, { dataset_id: string }>({
    mutationFn: (input) => postJSON<AiReportResponse>("/api/ai-report", input),
  });
}
