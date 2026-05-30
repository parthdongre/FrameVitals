import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface TargetColumnSelectProps {
  options: string[];
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

export function TargetColumnSelect({ options, value, onChange, disabled }: TargetColumnSelectProps) {
  return (
    <Card className="border-white/5 bg-white/[0.03] shadow-panel">
      <CardHeader className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <CardTitle className="text-base text-slate-50">Target Column</CardTitle>
          <Badge variant="outline" className="border-cyan-400/20 text-cyan-200">
            Optional
          </Badge>
        </div>
        <CardDescription className="text-slate-400">
          Select the response column for target analysis, feature importance, baseline modelling, and leakage checks.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <label className="block text-xs uppercase tracking-[0.28em] text-slate-500">
          Response variable
          <select
            value={value}
            onChange={(event) => onChange(event.target.value)}
            disabled={disabled}
            className="mt-2 h-12 w-full rounded-2xl border border-white/10 bg-space-900/75 px-4 text-sm text-slate-100 outline-none transition focus:border-cyan-400/40 focus:ring-2 focus:ring-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <option value="">Automatic / not selected</option>
            {options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <p className="text-xs leading-6 text-slate-500">
          If you do not select a target column, the dashboard will still run the general data quality and profiling analyses.
        </p>
      </CardContent>
    </Card>
  );
}
