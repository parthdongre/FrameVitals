import { useState } from "react";
import { motion } from "framer-motion";
import { Eyebrow } from "@/components/site/SiteShell";
import { EmptyState } from "@/components/ui/EmptyState";
import { Markdown } from "@/components/ui/Markdown";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { isPresent, safeObj, safeStr } from "@/lib/safe";
import { useAiReportMutation } from "@/hooks/useAiReportMutation";
import { cn } from "@/lib/utils";
import type { TabComponentProps } from "./tabRegistry";

/**
 * AI Report tab — renders the agent narrative as styled markdown.
 *
 * The pipeline skips the LLM call during /api/analyze by default (so the
 * report renders fast), so this tab also offers an on-demand "Generate now"
 * button that hits POST /api/ai-report and refreshes its own copy.
 *
 * Markdown rendering is delegated to `<Markdown>` so the same parser /
 * styling lives in one place — Ask Anything uses the same component.
 */
export default function AiReportTab({ analysis }: TabComponentProps) {
  const t = analysis as unknown as Record<string, unknown>;
  const initialAi = safeObj(t.aiReport, {} as Record<string, unknown>);
  const datasetId = safeStr((analysis as unknown as { id?: string }).id, "");

  // Local state lets us swap in the on-demand result without re-analyzing.
  const [ai, setAi] = useState(initialAi);
  const mutation = useAiReportMutation();

  const text = safeStr(ai.text, "");
  const source = safeStr(ai.source, "");
  const deferred = ai.deferred === true || source === "deferred";

  // Empty deferred state — show a Generate CTA instead of the empty state.
  if (deferred && !text) {
    return (
      <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-6">
        <motion.section variants={staggerChild} className="flex flex-wrap items-center gap-3">
          <Eyebrow>AI Narrative</Eyebrow>
          <SourceBadge source="deferred" />
        </motion.section>

        <motion.section
          variants={staggerChild}
          className="rounded-md border border-line bg-bg-1 p-8 text-center"
        >
          <p className="font-display text-[20px] leading-7 text-ink-1">
            AI report is on demand
          </p>
          <p className="mx-auto mt-2 max-w-md text-[13px] leading-6 text-ink-3">
            The LLM phase is skipped during analyze so the rest of the report renders fast.
            Click below to generate the narrative whenever you want — typically 5-15 seconds
            on local Ollama.
          </p>
          <button
            type="button"
            disabled={mutation.isPending || !datasetId}
            className="btn-accent mt-5 disabled:cursor-not-allowed disabled:opacity-40"
            onClick={async () => {
              if (!datasetId) return;
              const result = await mutation.mutateAsync({ dataset_id: datasetId });
              setAi(result as unknown as Record<string, unknown>);
            }}
          >
            {mutation.isPending ? "Generating…" : "Generate AI report"}
          </button>
          {mutation.error ? (
            <p className="mt-3 font-mono text-[11px] text-rose-300">
              {mutation.error.message}
            </p>
          ) : null}
        </motion.section>
      </motion.div>
    );
  }

  if (!isPresent(ai) || !text) {
    return (
      <EmptyState
        title="AI report not available for this dataset"
        hint="The local agent (Ollama) was not reachable, or it produced no narrative for this run. Re-run with Ollama running, or configure OPENROUTER_API_KEY for a remote fallback."
      />
    );
  }

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-6">
      <motion.section variants={staggerChild} className="flex flex-wrap items-center gap-3">
        <Eyebrow>AI Narrative</Eyebrow>
        <SourceBadge source={source || "unknown"} />
        {datasetId ? (
          <button
            type="button"
            onClick={async () => {
              const result = await mutation.mutateAsync({ dataset_id: datasetId });
              setAi(result as unknown as Record<string, unknown>);
            }}
            disabled={mutation.isPending}
            className="btn-ghost text-[12px] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {mutation.isPending ? "Regenerating…" : "Regenerate"}
          </button>
        ) : null}
      </motion.section>

      <motion.article
        variants={staggerChild}
        className="max-w-3xl rounded-md border border-line bg-bg-1 p-6"
      >
        <Markdown>{text}</Markdown>
      </motion.article>
    </motion.div>
  );
}

/* --------------------------------------------------------------------------
 * Source badge — small chip that signals where the narrative came from
 * (local Ollama / OpenRouter / heuristic fallback).
 * -------------------------------------------------------------------------- */

function SourceBadge({ source }: { source: string }) {
  const tone =
    source.includes("ollama") || source.includes("local")
      ? "ok"
      : source.includes("openrouter")
        ? "warn"
        : source.includes("fallback") || source.includes("error")
          ? "muted"
          : "neutral";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.24em]",
        tone === "ok" && "border-accent-line bg-accent-soft text-accent",
        tone === "warn" && "border-amber-300/30 bg-amber-300/10 text-amber-200",
        tone === "muted" && "border-line bg-bg-2 text-ink-3",
        tone === "neutral" && "border-line-strong bg-bg-2 text-ink-1",
      )}
    >
      <span aria-hidden className="h-1.5 w-1.5 rounded-full bg-current" />
      {source}
    </span>
  );
}
