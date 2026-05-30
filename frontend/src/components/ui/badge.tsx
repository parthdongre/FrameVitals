import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-supernova-400/20 bg-supernova-500/15 text-supernova-100",
        cyan: "border-supernova-400/20 bg-supernova-500/15 text-supernova-100",
        violet: "border-pulsar-500/20 bg-pulsar-500/15 text-pulsar-300",
        muted: "border-white/10 bg-white/[0.04] text-slate-300",
        outline: "border-white/10 bg-transparent text-slate-200",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };