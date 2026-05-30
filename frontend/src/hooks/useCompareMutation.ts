import { useMutation } from "@tanstack/react-query";
import { postFormData } from "@/lib/api";
import type { DriftReport } from "@/data/payload";

export interface CompareInput {
  reference: File;
  current: File;
  /**
   * Optional comma-separated list of columns to restrict the comparison.
   */
  columns?: string[];
}

export interface CompareSelfInput {
  dataset: File;
  date_column: string;
  /**
   * Split ratio between 0.1 and 0.9. Default 0.5.
   */
  ratio?: number;
}

/**
 * Compare two uploaded datasets. Backed by `POST /api/compare`.
 */
export function useCompareMutation() {
  return useMutation<DriftReport, Error, CompareInput>({
    mutationFn: ({ reference, current, columns }) => {
      const fd = new FormData();
      fd.append("reference", reference);
      fd.append("current", current);
      if (columns && columns.length) fd.append("columns", columns.join(","));
      return postFormData<DriftReport>("/api/compare", fd);
    },
  });
}

/**
 * Self-compare via a date split. Backed by `POST /api/compare-self`.
 */
export function useCompareSelfMutation() {
  return useMutation<DriftReport, Error, CompareSelfInput>({
    mutationFn: ({ dataset, date_column, ratio }) => {
      const fd = new FormData();
      fd.append("dataset", dataset);
      fd.append("date_column", date_column);
      if (typeof ratio === "number") fd.append("ratio", String(ratio));
      return postFormData<DriftReport>("/api/compare-self", fd);
    },
  });
}
