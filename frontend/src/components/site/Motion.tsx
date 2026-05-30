import { useEffect, useRef, useState, type ReactNode } from "react";
import { motion, useInView, useMotionValue, useScroll, useSpring, useTransform } from "framer-motion";
import { cn } from "@/lib/utils";

/* --------------------------------------------------------------------------
 * Reveal — fade + Y on scroll into view, with optional stagger delay
 * -------------------------------------------------------------------------- */
export function Reveal({
  children,
  delay = 0,
  y = 18,
  className,
  once = true,
}: {
  children: ReactNode;
  delay?: number;
  y?: number;
  className?: string;
  once?: boolean;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const inView = useInView(ref, { amount: 0.2, once });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y }}
      transition={{ duration: 0.65, delay, ease: "easeOut" }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* --------------------------------------------------------------------------
 * Stagger — animates children one after another
 * -------------------------------------------------------------------------- */
export function Stagger({
  children,
  step = 0.06,
  className,
}: {
  children: ReactNode[];
  step?: number;
  className?: string;
}) {
  return (
    <div className={className}>
      {children.map((child, i) => (
        <Reveal key={i} delay={i * step}>
          {child}
        </Reveal>
      ))}
    </div>
  );
}

/* --------------------------------------------------------------------------
 * CountUp — tween-animated number on first paint
 * -------------------------------------------------------------------------- */
export function CountUp({
  value,
  duration = 1.2,
  suffix = "",
  prefix = "",
  format = (v: number) => Math.round(v).toLocaleString(),
}: {
  value: number;
  duration?: number;
  suffix?: string;
  prefix?: string;
  format?: (v: number) => string;
}) {
  const [display, setDisplay] = useState<string>(format(0));
  const ref = useRef<HTMLSpanElement | null>(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });

  useEffect(() => {
    if (!inView) return;
    let raf = 0;
    const start = performance.now();
    const from = 0;
    const to = Number.isFinite(value) ? value : 0;
    const tick = (t: number) => {
      const elapsed = (t - start) / 1000;
      const k = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - k, 3); // easeOutCubic
      const v = from + (to - from) * eased;
      setDisplay(format(v));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, value, duration, format]);

  return (
    <span ref={ref} className="tnum">
      {prefix}
      {display}
      {suffix}
    </span>
  );
}

/* --------------------------------------------------------------------------
 * Tilt — subtle 3D tilt on hover, spring-physics damped
 * -------------------------------------------------------------------------- */
export function Tilt({
  children,
  className,
  max = 6,
}: {
  children: ReactNode;
  className?: string;
  max?: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const rotateX = useSpring(useTransform(y, [-0.5, 0.5], [max, -max]), { stiffness: 200, damping: 20 });
  const rotateY = useSpring(useTransform(x, [-0.5, 0.5], [-max, max]), { stiffness: 200, damping: 20 });

  return (
    <motion.div
      ref={ref}
      onMouseMove={(e) => {
        const rect = ref.current?.getBoundingClientRect();
        if (!rect) return;
        x.set((e.clientX - rect.left) / rect.width - 0.5);
        y.set((e.clientY - rect.top) / rect.height - 0.5);
      }}
      onMouseLeave={() => {
        x.set(0);
        y.set(0);
      }}
      style={{ rotateX, rotateY, transformStyle: "preserve-3d" }}
      className={cn("will-change-transform", className)}
    >
      {children}
    </motion.div>
  );
}

/* --------------------------------------------------------------------------
 * Spotlight — cursor-tracked radial gradient overlay
 * -------------------------------------------------------------------------- */
export function Spotlight({ className }: { className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const x = useMotionValue(-9999);
  const y = useMotionValue(-9999);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const rect = ref.current?.getBoundingClientRect();
      if (!rect) return;
      x.set(e.clientX - rect.left);
      y.set(e.clientY - rect.top);
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [x, y]);

  const background = useTransform(
    [x, y],
    ([cx, cy]) =>
      `radial-gradient(420px circle at ${cx}px ${cy}px, rgba(94,234,212,0.08), transparent 40%)`,
  );

  return (
    <motion.div
      ref={ref}
      style={{ background }}
      className={cn("pointer-events-none absolute inset-0", className)}
    />
  );
}

/* --------------------------------------------------------------------------
 * ScrollProgress — top horizontal bar that tracks page scroll
 * -------------------------------------------------------------------------- */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 20, restDelta: 0.001 });
  return (
    <motion.div
      style={{ scaleX, transformOrigin: "0% 50%" }}
      className="fixed inset-x-0 top-0 z-40 h-[2px] bg-[var(--accent)]"
    />
  );
}

/* --------------------------------------------------------------------------
 * Marquee — horizontal infinite scroll for chip rows
 * -------------------------------------------------------------------------- */
export function Marquee({ children, speed = 28 }: { children: ReactNode; speed?: number }) {
  return (
    <div className="overflow-hidden">
      <motion.div
        animate={{ x: ["0%", "-50%"] }}
        transition={{ duration: speed, ease: "linear", repeat: Infinity }}
        className="flex w-max gap-6"
      >
        {children}
        {children}
      </motion.div>
    </div>
  );
}

/* --------------------------------------------------------------------------
 * Parallax — translate Y on scroll, used for hero decoration
 * -------------------------------------------------------------------------- */
export function Parallax({
  children,
  range = 60,
  className,
}: {
  children: ReactNode;
  range?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start end", "end start"] });
  const y = useTransform(scrollYProgress, [0, 1], [range, -range]);
  return (
    <motion.div ref={ref} style={{ y }} className={className}>
      {children}
    </motion.div>
  );
}
