import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import type { AnalysisMode } from "@/data/mockTelemetry";

interface AnalysisProgressProps {
  progress: number;
  phase: string;
  fileName: string;
  analysisMode: AnalysisMode;
}

function formatAnalysisMode(mode: unknown, fallback = "STANDARD") {
  if (typeof mode !== "string" || !mode.trim()) {
    return fallback;
  }

  return mode.toUpperCase();
}

export function AnalysisProgress({ progress, phase, fileName, analysisMode }: AnalysisProgressProps) {
  return (
    <Card className="overflow-hidden border-cyan-400/15 bg-white/[0.03] shadow-panel">
      <CardHeader className="border-b border-white/5 bg-white/[0.015]">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <CardTitle className="text-slate-50">Analysis in Progress</CardTitle>
            <CardDescription className="mt-1 text-slate-400">
              {fileName} · {formatAnalysisMode(analysisMode)}
            </CardDescription>
          </div>
          <Badge variant="cyan" className="border-cyan-400/20 text-cyan-100">
            {Math.round(progress)}%
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 p-6">
        <div className="relative h-3 overflow-hidden rounded-full border border-white/5 bg-white/[0.04]">
          <motion.div
            className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-cyan-500 via-cyan-300 to-violet-500 shadow-[0_0_18px_rgba(6,182,212,0.5)]"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            transition={{ ease: "easeOut", duration: 0.2 }}
          />
          <motion.div
            className="absolute inset-y-0 left-0 w-1/3 bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.8),transparent)] opacity-35 blur-sm"
            animate={{ x: ["-120%", "320%"] }}
            transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
          />
        </div>

        <div className="flex items-center justify-between gap-3 text-xs uppercase tracking-[0.3em] text-slate-500">
          <span>{phase}</span>
          <span>Processing dataset and building the report</span>
        </div>
      </CardContent>
    </Card>
  );
}