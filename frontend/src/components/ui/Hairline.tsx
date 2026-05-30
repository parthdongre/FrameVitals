import { cn } from "@/lib/utils";

interface HairlineProps {
  className?: string;
  /**
   * Strong variant uses `--line-strong` for slightly more visible dividers
   * (typical between major sections). Default is the subtle `--line`.
   */
  strong?: boolean;
  /**
   * Vertical orientation flips the divider into a 1px column. Default is
   * horizontal.
   */
  vertical?: boolean;
}

/**
 * A 1px divider that always uses the editorial line tokens. No shadow, no
 * gradient — just a hairline. Used to separate sections without adding visual
 * noise.
 */
export function Hairline({ className, strong, vertical }: HairlineProps) {
  if (vertical) {
    return (
      <span
        aria-hidden="true"
        className={cn(
          "inline-block h-full w-px self-stretch",
          strong ? "bg-line-strong" : "bg-line",
          className,
        )}
      />
    );
  }
  return (
    <hr
      aria-hidden="true"
      className={cn(
        "h-px w-full border-0",
        strong ? "bg-line-strong" : "bg-line",
        className,
      )}
    />
  );
}
