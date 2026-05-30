import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { AnalysisMode } from "@/data/mockTelemetry";

const MODES: Array<{ value: AnalysisMode; label: string; hint: string }> = [
  { value: "quick", label: "Quick", hint: "Fast scan" },
  { value: "standard", label: "Standard", hint: "Recommended" },
  { value: "deep", label: "Deep", hint: "Full analysis" },
  { value: "research", label: "Research", hint: "Everything" },
];

interface AnalysisModeToggleProps {
  value: AnalysisMode;
  onChange: (mode: AnalysisMode) => void;
  disabled?: boolean;
}

export function AnalysisModeToggle({ value, onChange, disabled }: AnalysisModeToggleProps) {
  return (
    <div className="rounded-[1.45rem] border border-white/5 bg-white/[0.03] p-4 shadow-panel">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.34em] text-slate-500">Analysis Depth</p>
          <p className="mt-1 text-sm text-slate-300">Select how much of the analysis pipeline should run.</p>
        </div>
        <p className="text-[11px] uppercase tracking-[0.28em] text-cyan-300/70">Analysis modes</p>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {MODES.map((mode) => {
          const active = mode.value === value;

          return (
            <Button
              key={mode.value}
              type="button"
              variant={active ? "default" : "secondary"}
              disabled={disabled}
              onClick={() => onChange(mode.value)}
              className={cn(
                "h-auto flex-col items-start justify-start gap-1 rounded-2xl px-4 py-4 text-left",
                active ? "shadow-halo" : "",
              )}
            >
              <span className="text-sm font-semibold">{mode.label}</span>
              <span className="text-[11px] uppercase tracking-[0.26em] opacity-70">{mode.hint}</span>
            </Button>
          );
        })}
      </div>
    </div>
  );
}