import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "group relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-supernova-400/40 focus-visible:ring-offset-0 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "border border-supernova-400/15 bg-supernova-500 text-space-950 shadow-[0_0_18px_rgba(6,182,212,0.2)] hover:bg-supernova-400 hover:shadow-[0_0_28px_rgba(6,182,212,0.28)]",
        secondary:
          "border border-white/10 bg-white/[0.04] text-slate-100 hover:border-supernova-400/25 hover:bg-white/[0.06]",
        outline:
          "border border-supernova-400/20 bg-transparent text-supernova-100 hover:border-supernova-400/45 hover:bg-supernova-500/10",
        ghost: "bg-transparent text-slate-200 hover:bg-white/[0.04]",
        trace:
          "overflow-hidden border border-white/10 bg-white/[0.03] text-slate-50 hover:border-supernova-400/40 hover:bg-white/[0.05]",
        link: "bg-transparent p-0 text-supernova-300 underline-offset-4 hover:underline",
      },
      size: {
        default: "h-11 px-4 py-2",
        sm: "h-9 rounded-lg px-3 text-xs",
        lg: "h-12 rounded-2xl px-6 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, children, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";

    return (
      <Comp ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props}>
        {variant === "trace" ? (
          <>
            <span className="relative z-10">{children}</span>
            <svg
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 h-full w-full rounded-[inherit]"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              <rect
                x="1"
                y="1"
                width="98"
                height="98"
                rx="14"
                fill="none"
                stroke="rgba(103, 232, 249, 0.7)"
                strokeWidth="1"
                strokeDasharray="220"
                strokeDashoffset="220"
                className="transition-[stroke-dashoffset] duration-700 [stroke-dashoffset:220] group-hover:[stroke-dashoffset:0] group-hover:animate-trace"
              />
            </svg>
          </>
        ) : (
          children
        )}
      </Comp>
    );
  },
);

Button.displayName = "Button";

export { Button, buttonVariants };