import { useId, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface DisclosureProps {
  /**
   * Trigger label rendered next to the caret.
   */
  label: ReactNode;
  /**
   * Optional eyebrow above the label (mono uppercase).
   */
  eyebrow?: string;
  children: ReactNode;
  /**
   * Initial open state. Default false (collapsed).
   */
  defaultOpen?: boolean;
  className?: string;
}

/**
 * Accessible collapsible. Built on a real `<button aria-expanded>` for
 * screen-reader correctness. Used by `HowThisWorks` and any tab that wants
 * a "show details" affordance.
 */
export function Disclosure({
  label,
  eyebrow,
  children,
  defaultOpen = false,
  className,
}: DisclosureProps) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  const panelId = `disclosure-${id}`;

  return (
    <div
      className={cn(
        "rounded-md border border-line bg-bg-1 transition-colors",
        open ? "border-line-strong" : "hover:border-line-strong",
        className,
      )}
    >
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left"
      >
        <span className="flex flex-col">
          {eyebrow ? (
            <span className="font-mono text-[10px] uppercase tracking-[0.32em] text-ink-3">
              {eyebrow}
            </span>
          ) : null}
          <span className="text-[14px] text-ink-1">{label}</span>
        </span>
        <ChevronDown
          aria-hidden="true"
          className={cn(
            "h-4 w-4 shrink-0 text-ink-3 transition-transform duration-200",
            open && "rotate-180 text-ink-1",
          )}
        />
      </button>
      {open ? (
        <div
          id={panelId}
          className="border-t border-line px-4 py-4 text-[13px] leading-6 text-ink-2"
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}
