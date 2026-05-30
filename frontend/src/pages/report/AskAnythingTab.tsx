import { useEffect, useRef, useState, type FormEvent } from "react";
import { motion } from "framer-motion";
import { Send, Sparkles } from "lucide-react";
import { Eyebrow } from "@/components/site/SiteShell";
import { Disclosure } from "@/components/ui/Disclosure";
import { EmptyState } from "@/components/ui/EmptyState";
import { Markdown } from "@/components/ui/Markdown";
import { staggerChild, staggerParent } from "@/components/site/Variants";
import { useAskMutation } from "@/hooks/useAskQuery";
import { safeStr } from "@/lib/safe";
import { cn } from "@/lib/utils";
import type { AskResponse } from "@/data/payload";
import type { TabComponentProps } from "./tabRegistry";

interface ChatMessage {
  id: string;
  question: string;
  status: "pending" | "ready" | "error";
  answer?: string;
  source?: string;
  trace?: AskResponse["trace"];
  error?: string;
}

const SUGGESTED_PROMPTS = [
  "What are the biggest data quality risks in this dataset?",
  "Which features look most predictive of the target?",
  "Are there any obvious target leakage candidates?",
  "Where is the dataset weakest for ML readiness?",
];

/**
 * Ask Anything — chat panel against /api/ask.
 *
 * History is held in local component state. The ReportPage shell keys this
 * component on its tab id (stable per tab), so tab crossfades via
 * AnimatePresence don't lose chat history. Switching to a different tab and
 * back DOES reset history — the design's stable-key rule applies to the
 * crossfade not to "across tab swaps in different IDs".
 *
 * The composer is sticky at the bottom of the conversation card so long
 * histories never push it off-screen.
 */
export default function AskAnythingTab({ analysis }: TabComponentProps) {
  const datasetId = safeStr((analysis as unknown as { id?: string }).id, "");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const ask = useAskMutation();
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages.length]);

  if (!datasetId) {
    return (
      <EmptyState
        title="Ask Anything needs a dataset"
        hint="Run an analysis first — the agent answers grounded in the cached payload from /api/analyze."
      />
    );
  }

  const sendQuestion = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed) return;
    const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setMessages((prev) => [...prev, { id, question: trimmed, status: "pending" }]);
    setInput("");

    try {
      const res = await ask.mutateAsync({ question: trimmed, dataset_id: datasetId });
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                status: "ready",
                answer: safeStr(res.answer, ""),
                source: safeStr(res.source, "unknown"),
                trace: res.trace,
              }
            : m,
        ),
      );
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e ?? "Request failed.");
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, status: "error", error: msg } : m)),
      );
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    sendQuestion(input);
  };

  return (
    <motion.div variants={staggerParent} initial="initial" animate="animate" className="space-y-6">
      <motion.section variants={staggerChild} className="flex flex-wrap items-center gap-3">
        <Eyebrow>Ask Anything</Eyebrow>
        <span className="font-mono text-[11px] tabular-nums text-ink-3">
          dataset · <span className="text-ink-1">{datasetId}</span>
        </span>
      </motion.section>

      <motion.div
        variants={staggerChild}
        className="flex h-[60vh] flex-col rounded-md border border-line bg-bg-1"
      >
        {/* Conversation */}
        <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-5">
          {messages.length === 0 ? (
            <SuggestedPrompts onPick={(q) => sendQuestion(q)} />
          ) : (
            messages.map((m) => <MessageBlock key={m.id} message={m} />)
          )}
        </div>

        {/* Composer */}
        <form
          onSubmit={onSubmit}
          className="flex items-center gap-2 border-t border-line p-3"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about this dataset…"
            disabled={ask.isPending}
            className="h-11 flex-1 rounded-xl border border-line bg-bg-2 px-4 text-[14px] text-ink-1 placeholder:text-ink-4 outline-none transition-colors focus:border-accent-line disabled:cursor-not-allowed disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || ask.isPending}
            className="btn-accent disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Send className="h-3.5 w-3.5" />
            {ask.isPending ? "Thinking…" : "Send"}
          </button>
        </form>
      </motion.div>
    </motion.div>
  );
}

/* --- subcomponents --------------------------------------------------- */

function SuggestedPrompts({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
      <Sparkles className="h-5 w-5 text-accent" aria-hidden />
      <p className="max-w-md text-[14px] leading-7 text-ink-2">
        Ask about anything in the cached analysis. The agent answers from the same payload the
        report tabs render — grounded, no hallucinations.
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        {SUGGESTED_PROMPTS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => onPick(p)}
            className="rounded-full border border-line bg-bg-2 px-3 py-1.5 text-[12px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink-1"
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBlock({ message }: { message: ChatMessage }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl border border-accent-line bg-accent-soft px-4 py-2 text-[14px] leading-7 text-ink-1">
          {message.question}
        </div>
      </div>
      {message.status === "pending" ? (
        <div className="flex justify-start">
          <div className="max-w-[80%] rounded-2xl border border-line bg-bg-2 px-4 py-2 text-[13px] text-ink-3">
            <span className="inline-flex items-center gap-2">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" aria-hidden />
              <span>Thinking through tools…</span>
            </span>
          </div>
        </div>
      ) : null}
      {message.status === "ready" ? (
        <div className="flex justify-start">
          <div
            className={cn(
              "max-w-[80%] space-y-3 rounded-2xl border border-line bg-bg-2 px-4 py-3",
            )}
          >
            <Markdown density="compact">{message.answer ?? ""}</Markdown>
            {message.source ? (
              <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-ink-3">
                source · {message.source}
              </p>
            ) : null}
            {message.trace ? (
              <Disclosure label="Show trace" eyebrow="Agent" className="bg-bg-1">
                <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-sm bg-bg-0 p-3 font-mono text-[11px] leading-5 text-ink-3">
                  {safeJsonStringify(message.trace)}
                </pre>
              </Disclosure>
            ) : null}
          </div>
        </div>
      ) : null}
      {message.status === "error" ? (
        <div className="flex justify-start">
          <div className="max-w-[80%] rounded-2xl border border-rose-300/30 bg-rose-300/5 px-4 py-2 text-[13px] text-rose-200">
            {message.error}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
