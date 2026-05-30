import { motion } from "framer-motion";
import { Database, FileText, Layers3, Radar, Satellite } from "lucide-react";
import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { DashboardTelemetry } from "@/data/mockTelemetry";

interface SidebarTelemetryProps {
  telemetry: DashboardTelemetry;
  fileName: string;
  selectedFile: File | null;
  commandEcho: string;
}

function formatBytes(value: number) {
  const units = ["B", "KB", "MB", "GB"];
  let current = value;
  let index = 0;

  while (current >= 1024 && index < units.length - 1) {
    current /= 1024;
    index += 1;
  }

  return `${current.toFixed(current >= 10 ? 0 : 1)} ${units[index]}`;
}

function formatCount(value: unknown, fallback = "0") {
  const numericValue = typeof value === "number" ? value : Number(value);

  if (!Number.isFinite(numericValue)) {
    return fallback;
  }

  return numericValue.toLocaleString();
}

function formatAnalysisMode(mode: unknown, fallback = "STANDARD") {
  if (typeof mode !== "string" || !mode.trim()) {
    return fallback;
  }

  return mode.toUpperCase();
}

function safeArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

export function SidebarTelemetry({ telemetry, fileName, selectedFile, commandEcho }: SidebarTelemetryProps) {
  const targetCandidates = safeArray<string>(telemetry.targetCandidates ?? telemetry.rolesSummary?.target_candidates);

  return (
    <div className="flex h-full flex-col gap-4 text-slate-100">
      <motion.div
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.35 }}
        className="rounded-3xl border border-white/5 bg-white/[0.03] p-5 shadow-panel"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-[11px] uppercase tracking-[0.35em] text-cyan-300/70">Analysis dashboard</p>
            <h2 className="mt-2 text-xl font-semibold text-slate-50">Dataset summary</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Key dataset metadata, analysis context, and download links are pinned here for quick review.
            </p>
          </div>
          <div className="rounded-2xl border border-cyan-400/15 bg-cyan-500/10 p-3 text-cyan-200 shadow-halo">
            <Satellite className="h-5 w-5" />
          </div>
        </div>

        <div className="mt-5 space-y-3 rounded-2xl border border-white/5 bg-space-900/60 p-4">
          <TelemetryRow icon={<FileText className="h-4 w-4" />} label="Filename" value={fileName} />
          <TelemetryRow icon={<Database className="h-4 w-4" />} label="Rows" value={formatCount(telemetry.rows)} mono />
          <TelemetryRow icon={<Layers3 className="h-4 w-4" />} label="Columns" value={formatCount(telemetry.columns)} mono />
          <TelemetryRow icon={<Radar className="h-4 w-4" />} label="Analysis mode" value={formatAnalysisMode(telemetry.analysisMode)} />
          <TelemetryRow
            icon={<Radar className="h-4 w-4" />}
            label="Target column"
            value={telemetry.selectedTargetColumn || "Not selected"}
          />
        </div>
      </motion.div>

      <Card className="border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader>
          <CardTitle className="text-sm uppercase tracking-[0.35em] text-slate-400">Data types</CardTitle>
          <CardDescription>Column classes detected in the current dataset.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {safeArray<any>(telemetry.dataTypes).map((item) => (
            <Badge key={item.label} variant="outline" className="border-white/10 text-slate-200">
              {item.label}: {item.count}
            </Badge>
          ))}
        </CardContent>
      </Card>

      <Card className="border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader>
          <CardTitle className="text-sm uppercase tracking-[0.35em] text-slate-400">Dataset metrics</CardTitle>
          <CardDescription>{telemetry.fileSize} file size | last analysis {telemetry.lastScan}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {safeArray<any>(telemetry.metrics).map((metric) => (
            <div key={metric.label} className="rounded-2xl border border-white/5 bg-white/[0.02] p-3">
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm text-slate-300">{metric.label}</p>
                <p className="font-mono text-sm text-cyan-200">{metric.value}</p>
              </div>
              <p className="mt-1 text-xs text-slate-500">{metric.hint}</p>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card className="border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader>
          <CardTitle className="text-sm uppercase tracking-[0.35em] text-slate-400">Command output</CardTitle>
          <CardDescription>Latest dataset analysis prompt or status update.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-2xl border border-cyan-400/10 bg-cyan-500/5 p-4 text-sm leading-6 text-slate-300">
            {commandEcho}
          </div>
          <p className="mt-3 text-xs uppercase tracking-[0.28em] text-slate-500">
            {selectedFile ? `File size ${formatBytes(selectedFile.size)} loaded` : "Awaiting file upload"}
          </p>
        </CardContent>
      </Card>

      <Card className="border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader>
          <CardTitle className="text-sm uppercase tracking-[0.35em] text-slate-400">Target candidates</CardTitle>
          <CardDescription>Columns suitable for target-based analyses.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {targetCandidates.length > 0 ? (
            targetCandidates.map((candidate: string) => (
              <Badge key={candidate} variant="outline" className="border-white/10 text-slate-200">
                {candidate}
              </Badge>
            ))
          ) : (
            <p className="text-sm text-slate-500">No target candidates available yet.</p>
          )}
        </CardContent>
      </Card>

      <div className="mt-auto rounded-3xl border border-white/5 bg-space-900/50 p-4 text-xs text-slate-500">
        <p className="uppercase tracking-[0.35em] text-slate-400">System status</p>
        <p className="mt-2 leading-6">
          The analysis environment is synchronized. Data quality checks are running normally.
        </p>
        {telemetry.downloadLinks ? (
          <div className="mt-4 flex flex-col gap-2">
            {telemetry.downloadLinks.reportReady === false ? (
              <div className="rounded-2xl border border-amber-400/15 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
                PDF report is generating in the background.
              </div>
            ) : (
              <Button asChild variant="secondary" size="sm" className="justify-start">
                <a href={telemetry.downloadLinks.report}>Download PDF Report</a>
              </Button>
            )}
            <Button asChild variant="secondary" size="sm" className="justify-start">
              <a href={telemetry.downloadLinks.cleaned}>Download Cleaned CSV</a>
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function TelemetryRow({
  icon,
  label,
  value,
  mono = false,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/5 py-3 last:border-b-0">
      <div className="flex items-center gap-3 text-sm text-slate-400">
        <span className="text-cyan-300">{icon}</span>
        <span>{label}</span>
      </div>
      <p className={mono ? "font-mono text-sm text-slate-100" : "text-sm text-slate-100"}>{value}</p>
    </div>
  );
}