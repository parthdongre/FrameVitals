import { useMutation, type UseMutationOptions } from "@tanstack/react-query";
import { postJSON } from "@/lib/api";
import type { AskResponse } from "@/data/payload";

export interface AskInput {
  question: string;
  dataset_id: string;
}

/**
 * `/api/ask` mutation. Wraps the agentic Q&A endpoint so chat panels can
 * fire individual questions without a query cache. Each question becomes a
 * distinct mutation invocation; chat history is held in component state.
 */
export function useAskMutation(
  options?: UseMutationOptions<AskResponse, Error, AskInput>,
) {
  return useMutation<AskResponse, Error, AskInput>({
    mutationFn: (input) => postJSON<AskResponse>("/api/ask", input),
    ...(options ?? {}),
  });
}

// Backwards-compatible alias matching the design's filename.
export { useAskMutation as useAskQuery };
