import * as React from "react";
import { cn } from "@/lib/utils";

function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl bg-white/[0.04] before:absolute before:inset-0 before:animate-scan before:bg-[linear-gradient(90deg,transparent,rgba(6,182,212,0.24),transparent)] before:bg-[length:220%_100%]",
        className,
      )}
      {...props}
    />
  );
}

export { Skeleton };