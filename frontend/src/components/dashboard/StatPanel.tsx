import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { AnimatedCounter } from "@/components/dashboard/AnimatedCounter";

interface StatPanelProps {
  label: string;
  value: string;
  hint: string;
  accent?: "cyan" | "violet" | "slate";
  numericValue?: number;
  precision?: number;
}

const accentStyles: Record<
  NonNullable<StatPanelProps["accent"]>,
  { halo: string; text: string }
> = {
  cyan: { halo: "from-cyan-500/18 via-cyan-500/8 to-transparent", text: "text-cyan-100" },
  violet: { halo: "from-violet-500/18 via-violet-500/8 to-transparent", text: "text-violet-100" },
  slate: { halo: "from-white/8 via-white/4 to-transparent", text: "text-slate-100" },
};

export function StatPanel({ label, value, hint, accent = "cyan", numericValue, precision = 0 }: StatPanelProps) {
  return (
    <motion.div whileHover={{ y: -2 }} transition={{ duration: 0.18 }}>
      <Card className={cn("border-white/5 bg-white/[0.03] shadow-panel", accentStyles[accent].text)}>
        <div className={cn("absolute inset-0 bg-gradient-to-br opacity-100", accentStyles[accent].halo)} />
        <CardContent className="relative p-5">
          <p className="text-[11px] uppercase tracking-[0.35em] text-slate-500">{label}</p>
          <p className="mt-3 font-mono text-3xl tracking-tight text-slate-50">
            {typeof numericValue === "number" ? (
              <AnimatedCounter value={numericValue} precision={precision} />
            ) : (
              value
            )}
          </p>
          <p className="mt-2 text-sm text-slate-400">{hint}</p>
        </CardContent>
      </Card>
    </motion.div>
  );
}