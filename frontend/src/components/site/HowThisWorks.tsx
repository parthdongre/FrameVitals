import type { ReactNode } from "react";
import { Disclosure } from "@/components/ui/Disclosure";

export interface HowThisWorksContent {
  /**
   * Trigger label, e.g. "How this tab works".
   */
  title?: string;
  /**
   * Plain-language explanation.
   */
  body: ReactNode;
  /**
   * Optional algorithm chips (rendered as comma-separated mono labels).
   */
  algorithms?: string[];
  /**
   * Source path for the underlying Python module so power users can find it.
   */
  source?: string;
}

interface HowThisWorksProps extends HowThisWorksContent {
  className?: string;
  defaultOpen?: boolean;
}

/**
 * Inline disclosure rendered at the bottom of every report sub-tab. Keeps the
 * "what is this number" answer one click away, matching the design's
 * editorial-but-honest tone.
 */
export function HowThisWorks({
  title = "How this works",
  body,
  algorithms,
  source,
  className,
  defaultOpen,
}: HowThisWorksProps) {
  return (
    <Disclosure
      className={className}
      eyebrow="Method"
      label={title}
      defaultOpen={defaultOpen}
    >
      <div className="space-y-3">
        <div className="text-[13px] leading-6 text-ink-2">{body}</div>
        {algorithms && algorithms.length > 0 ? (
          <p className="font-mono text-[11px] tracking-wide text-ink-3">
            <span className="uppercase tracking-[0.2em] text-ink-4">Algorithms · </span>
            {algorithms.join(" · ")}
          </p>
        ) : null}
        {source ? (
          <p className="font-mono text-[11px] tracking-wide text-ink-3">
            <span className="uppercase tracking-[0.2em] text-ink-4">Source · </span>
            <code className="rounded-sm bg-bg-2 px-1.5 py-0.5 text-ink-2">{source}</code>
          </p>
        ) : null}
      </div>
    </Disclosure>
  );
}
