import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Eyebrow, Hr, PageTitle, Section, SectionHeader } from "@/components/site/SiteShell";
import { useAnalyzeDatasetMutation } from "@/hooks/useAnalyzeDatasetMutation";
import type { AnalysisMode, DashboardTelemetry } from "@/data/mockTelemetry";
import type { Route } from "@/components/site/SiteShell";
import { setAnalysis } from "@/lib/analysisStore";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

interface AnalyzePageProps {
  onResult: (result: DashboardTelemetry) => void;
  onNavigate: (route: Route) => void;
}

const MODES: { id: AnalysisMode; label: string; description: string }[] = [
  { id: "quick",    label: "Quick",    description: "Profiling and signals only — under a second on most datasets." },
  { id: "standard", label: "Standard", description: "Full v3 pipeline including deep stats, anomaly ensemble, ML leaderboard, and SHAP." },
  { id: "deep",     label: "Deep",     description: "Same as standard, with extra bivariate budgets and exhaustive chart generation." },
  { id: "research", label: "Research", description: "Maximum depth — slower, intended for research-grade reports." },
];

export function AnalyzePage({ onResult, onNavigate }: AnalyzePageProps) {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<AnalysisMode>("standard");
  const [target, setTarget] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [previewColumns, setPreviewColumns] = useState<string[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const mutation = useAnalyzeDatasetMutation();

  const onFile = async (f: File) => {
    setFile(f);
    setError(null);
    try {
      if (/\.csv$|\.tsv$/i.test(f.name)) {
        const buffer = await f.slice(0, 8 * 1024).text();
        const sep = f.name.endsWith(".tsv") ? "\t" : ",";
        const firstLine = buffer.split(/\r?\n/)[0] ?? "";
        const cols = firstLine
          .split(sep)
          .map((c) => c.trim().replace(/^"|"$/g, ""))
          .filter(Boolean)
          .slice(0, 64);
        setPreviewColumns(cols);
      } else {
        setPreviewColumns([]);
      }
    } catch {
      setPreviewColumns([]);
    }
  };

  const submit = async () => {
    if (!file) {
      setError("Choose a dataset first.");
      return;
    }
    setError(null);
    try {
      const result = await mutation.mutateAsync({
        file,
        analysisMode: mode,
        targetColumn: target || undefined,
      });
      // Defensive: result must carry the structural fields the report needs.
      // If id / rows / columns are missing the backend either crashed mid-run
      // or returned a placeholder; either way we shouldn't navigate to a
      // report that will render all zeros.
      if (
        !result ||
        typeof result !== "object" ||
        !(result as { id?: string }).id ||
        typeof (result as { rows?: unknown }).rows !== "number" ||
        typeof (result as { columns?: unknown }).columns !== "number"
      ) {
        const serverError =
          (result as { error?: string } | null)?.error ?? null;
        throw new Error(
          serverError ??
            "Server returned an unexpected response shape. Check the Flask logs.",
        );
      }
      // Hand the analysis to both the local-state route hand-off (for now)
      // and the module-level store (for future tabs that read from it).
      setAnalysis(result);
      onResult(result);
      onNavigate("report");
    } catch (e: any) {
      setError(e?.message ?? "Analysis failed.");
    }
  };

  const isPending = mutation.isPending;

  return (
    <>
      <Section className="pb-10 pt-6">
        <Eyebrow>Analyze</Eyebrow>
        <PageTitle subtitle="Upload a dataset, choose how deep to go, and pick a target column when you want machine-learning outputs. Re-running with the same inputs is cached.">
          Bring your data.
        </PageTitle>
      </Section>

      <Section className="py-0">
        <SectionHeader
          eyebrow="01 · Dataset"
          title="Upload"
          description="CSV, TSV, JSON, or Excel. Files stay on your machine."
        />

        <motion.button
          whileHover={{ borderColor: "var(--ink-2)" }}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) onFile(f);
          }}
          className={cn(
            "mt-4 flex w-full flex-col items-center justify-center gap-3 rounded-2xl border border-dashed bg-[var(--bg-1)] px-6 py-16 text-center transition-colors duration-200",
            dragOver
              ? "border-[var(--accent)]/50"
              : file
              ? "border-emerald-300/30"
              : "border-[var(--line-strong)]",
          )}
        >
          <span className="label-mono">{file ? "Ready" : "Drop file here"}</span>
          <span className="text-xl font-semibold text-[var(--ink-1)]">
            {file ? file.name : "Click to browse, or drag a file"}
          </span>
          <span className="text-xs text-[var(--ink-3)]">
            {file ? formatBytes(file.size) : "CSV · TSV · JSON · XLSX · XLS"}
          </span>
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.tsv,.xlsx,.xls,.json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onFile(f);
            }}
          />
        </motion.button>
      </Section>

      <Hr />

      <Section className="py-0">
        <SectionHeader
          eyebrow="02 · Analysis depth"
          title="Pick a mode"
          description="Quick is profiling-only. Standard runs the full v3 pipeline. Deep and Research add longer-running analytics."
        />

        <div className="grid gap-3 sm:grid-cols-2">
          {MODES.map((m) => {
            const active = mode === m.id;
            return (
              <motion.label
                key={m.id}
                whileHover={{ y: -2 }}
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
                className={cn(
                  "cursor-pointer rounded-2xl border p-5 transition-colors duration-200",
                  active
                    ? "border-[var(--accent-line)] bg-[var(--accent-soft)]"
                    : "border-[var(--line)] bg-[var(--bg-1)] hover:border-[var(--line-strong)]",
                )}
              >
                <input
                  type="radio"
                  className="hidden"
                  name="mode"
                  value={m.id}
                  checked={active}
                  onChange={() => setMode(m.id)}
                />
                <div className="flex items-center justify-between">
                  <span className="text-base font-semibold text-[var(--ink-1)]">{m.label}</span>
                  {active ? <span className="label-mono text-[var(--accent)]">Selected</span> : null}
                </div>
                <p className="mt-2 text-sm leading-7 text-[var(--ink-2)]">{m.description}</p>
              </motion.label>
            );
          })}
        </div>
      </Section>

      <Hr />

      <Section className="py-0">
        <SectionHeader
          eyebrow="03 · Target column (optional)"
          title="Unlock the ML lab"
          description="Pick a column to predict. The system will infer classification vs regression, train a 7-8 model leaderboard, and produce SHAP explanations for the winner."
        />

        {previewColumns.length > 0 ? (
          <div>
            <div className="-mx-6 mb-3 flex gap-2 overflow-x-auto px-6 sm:mx-0 sm:px-0">
              <ChipButton onClick={() => setTarget("")} active={target === ""}>None</ChipButton>
              {previewColumns.map((c) => (
                <ChipButton key={c} onClick={() => setTarget(c)} active={target === c}>
                  {c}
                </ChipButton>
              ))}
            </div>
            <p className="text-xs text-[var(--ink-3)]">
              Detected {previewColumns.length} columns from the file header. Or type one manually below.
            </p>
          </div>
        ) : null}

        <input
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          placeholder="Optional column name"
          className="mt-3 h-12 w-full max-w-md rounded-xl border border-[var(--line)] bg-[var(--bg-1)] px-4 text-sm text-[var(--ink-1)] outline-none transition focus:border-[var(--accent-line)]"
        />
      </Section>

      <Hr />

      <Section className="py-0">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={submit}
            disabled={!file || isPending}
            className={cn("btn-primary disabled:cursor-not-allowed disabled:opacity-40")}
          >
            {isPending ? "Running pipeline…" : "Run analysis →"}
          </button>
          <AnimatePresence>
            {error ? (
              <motion.span
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="text-sm text-rose-300"
              >
                {error}
              </motion.span>
            ) : isPending ? (
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-sm text-[var(--ink-3)]"
              >
                Phases run in parallel — typically 1–8 seconds depending on dataset size.
              </motion.span>
            ) : null}
          </AnimatePresence>
        </div>

        <AnimatePresence>
          {isPending ? (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="mt-6 overflow-hidden"
            >
              <ProgressBar />
            </motion.div>
          ) : null}
        </AnimatePresence>
      </Section>
    </>
  );
}

function ChipButton({
  children,
  active,
  onClick,
}: {
  children: React.ReactNode;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "shrink-0 rounded-full border px-3 py-1.5 text-xs transition",
        active
          ? "border-[var(--accent-line)] bg-[var(--accent-soft)] text-[var(--ink-1)]"
          : "border-[var(--line)] text-[var(--ink-3)] hover:border-[var(--line-strong)] hover:text-[var(--ink-1)]",
      )}
    >
      {children}
    </button>
  );
}

/**
 * Indeterminate animated progress bar with an elapsed counter and a phase
 * tagline that rotates based on real elapsed milliseconds. The tagline
 * sequence mirrors the actual pipeline order in `modules/pipeline.py` so the
 * user gets meaningful "what we are doing now" feedback instead of a generic
 * spinner.
 */
function ProgressBar() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const id = window.setInterval(() => {
      setElapsed(performance.now() - start);
    }, 100);
    return () => window.clearInterval(id);
  }, []);

  const phase = pickPhase(elapsed);
  const seconds = elapsed / 1000;

  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--bg-1)] p-4">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <p className="label-mono">Live</p>
        <p className="font-mono text-[11px] tabular-nums text-[var(--ink-3)]">
          {seconds.toFixed(1)} s
        </p>
      </div>
      <div className="relative h-1.5 overflow-hidden rounded-full bg-white/[0.04]">
        <motion.div
          className="absolute h-full rounded-full bg-[var(--accent)]"
          initial={{ x: "-30%", width: "30%" }}
          animate={{ x: "100%" }}
          transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>
      <AnimatePresence mode="wait">
        <motion.p
          key={phase.id}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25 }}
          className="mt-3 text-xs leading-6 text-[var(--ink-3)]"
        >
          <span className="font-mono uppercase tracking-[0.2em] text-[var(--ink-2)]">
            {phase.label}
          </span>
          <span> · {phase.detail}</span>
        </motion.p>
      </AnimatePresence>
    </div>
  );
}

interface AnalyzePhase {
  id: string;
  label: string;
  detail: string;
  /** Lower-bound ms after which this phase becomes the displayed tagline. */
  startMs: number;
}

const ANALYZE_PHASES: AnalyzePhase[] = [
  { id: "load",     label: "Loading",      detail: "reading the file and inferring dtypes",                       startMs: 0     },
  { id: "profile",  label: "Profiling",    detail: "shape, missingness, duplicates, correlations",                 startMs: 600   },
  { id: "stats",    label: "Statistics",   detail: "normality, distribution fits, bivariate effect sizes",         startMs: 1500  },
  { id: "anomaly",  label: "Anomalies",    detail: "seven-detector ensemble for agreement-based flagging",         startMs: 3000  },
  { id: "ml",       label: "ML lab",       detail: "5-fold CV across baseline, tree, boosted, and linear models",  startMs: 4500  },
  { id: "shap",     label: "SHAP",         detail: "global + per-row attributions for the leaderboard winner",     startMs: 7000  },
  { id: "ts",       label: "Time-series",  detail: "STL decomposition, ADF/KPSS, ACF/PACF, Holt-Winters forecast", startMs: 9000  },
  { id: "text",     label: "Text profile", detail: "TF-IDF + LSA, n-grams, regex pattern hits",                    startMs: 11000 },
  { id: "cleaning", label: "Cleaning",     detail: "imputation, dedupe, outlier clipping, health rescore",         startMs: 13000 },
  { id: "ai",       label: "AI report",    detail: "grounded narrative via local agent",                           startMs: 15000 },
];

function pickPhase(elapsedMs: number): AnalyzePhase {
  let current = ANALYZE_PHASES[0];
  for (const p of ANALYZE_PHASES) {
    if (elapsedMs >= p.startMs) current = p;
    else break;
  }
  return current;
}
