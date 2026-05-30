import { useQuery } from "@tanstack/react-query";
import { mockTelemetry } from "@/data/mockTelemetry";

export function useTelemetryQuery() {
  return useQuery({
    queryKey: ["dashboard-telemetry"],
    queryFn: async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 900));
      return mockTelemetry;
    },
  });
}