import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Search, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface CommandInputProps {
  onExecute?: (command: string) => void;
  missionLabel?: string;
  className?: string;
}

const PROMPTS = [
  "Ask a question about this dataset (e.g., 'What caused the Q3 revenue drop?')",
  "Ask about missing values, feature importance, or target leakage.",
  "Ask which columns need cleaning before modelling.",
  "Ask for a summary of the most important quality issues.",
];

export function CommandInput({ onExecute, missionLabel = "Data Assistant", className }: CommandInputProps) {
  const [command, setCommand] = useState("");
  const [placeholderIndex, setPlaceholderIndex] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setPlaceholderIndex((current) => (current + 1) % PROMPTS.length);
    }, 2600);

    return () => window.clearInterval(timer);
  }, []);

  const placeholder = PROMPTS[placeholderIndex];

  const submitCommand = () => {
    const trimmed = command.trim();
    if (trimmed.length === 0) {
      return;
    }
    onExecute?.(trimmed);
    setCommand("");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={cn(
        "sticky top-4 z-30 rounded-[1.6rem] border border-cyan-400/15 bg-white/[0.03] shadow-[0_0_0_1px_rgba(255,255,255,0.03),0_16px_44px_rgba(0,0,0,0.35)] backdrop-blur-2xl",
        className,
      )}
    >
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-cyan-400/15 bg-cyan-500/10 text-cyan-200 shadow-halo">
            <Sparkles className="h-4 w-4" />
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-[0.32em] text-slate-500">Command Bar</p>
            <p className="text-sm font-semibold text-slate-100">{missionLabel} ready</p>
          </div>
        </div>
        <div className="hidden items-center gap-2 text-[11px] uppercase tracking-[0.3em] text-cyan-300/70 sm:flex">
          <span className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_14px_rgba(6,182,212,0.8)]" />
          keyboard shortcut ready
        </div>
      </div>

      <div className="p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              submitCommand();
            }}
            className="flex flex-1 flex-col gap-3 lg:flex-row lg:items-center"
          >
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <Input
                value={command}
                onChange={(event) => setCommand(event.target.value)}
                placeholder={placeholder}
                className="h-12 border-white/10 bg-space-900/70 pl-11 font-mono text-[13px] text-slate-100 placeholder:text-slate-500 focus-visible:border-cyan-400/40 focus-visible:ring-cyan-400/25"
              />
              <div className="pointer-events-none absolute inset-y-0 right-3 hidden items-center sm:flex">
                <AnimatePresence mode="wait" initial={false}>
                  <motion.span
                    key={placeholder}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.18 }}
                    className="rounded-full border border-white/5 bg-white/[0.04] px-2.5 py-1 text-[10px] uppercase tracking-[0.26em] text-slate-400"
                  >
                    dataset query
                  </motion.span>
                </AnimatePresence>
              </div>
            </div>

            <Button type="submit" variant="trace" size="lg" className="min-w-36">
              Submit
            </Button>
          </form>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] uppercase tracking-[0.28em] text-slate-500">
          <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1">Quick analysis</span>
          <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1">Deep analysis</span>
          <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1">Data quality review</span>
        </div>
      </div>
    </motion.div>
  );
}