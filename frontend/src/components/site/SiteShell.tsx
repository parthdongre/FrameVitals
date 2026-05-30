import { Component, useEffect, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/utils";

export type Route = "home" | "analyze" | "report" | "charts" | "modules" | "about";

const ROUTES: { id: Route; label: string }[] = [
  { id: "home", label: "Overview" },
  { id: "analyze", label: "Analyze" },
  { id: "report", label: "Report" },
  { id: "charts", label: "Charts" },
  { id: "modules", label: "Modules" },
  { id: "about", label: "About" },
];

interface SiteShellProps {
  route: Route;
  onNavigate: (route: Route) => void;
  children: ReactNode;
}

/**
 * Editorial shell, warm cream on near-black. Inspired by deepfake-analyzer-nine.
 */
export function SiteShell({ route, onNavigate, children }: SiteShellProps) {
  return (
    <div className="relative min-h-screen text-[15px] leading-relaxed">
      <TopNav route={route} onNavigate={onNavigate} />

      <ErrorBoundary>
        <AnimatePresence mode="wait" initial={false}>
          <motion.main
            key={route}
            initial={{ opacity: 0, y: 14, filter: "blur(2px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -8, filter: "blur(2px)" }}
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="mx-auto w-full max-w-6xl px-6 pb-32 pt-10 sm:px-10 lg:px-16"
          >
            {children}
          </motion.main>
        </AnimatePresence>
      </ErrorBoundary>

      <Footer />
    </div>
  );
}

/* ----------------------------- Top Nav ----------------------------------- */

function TopNav({ route, onNavigate }: { route: Route; onNavigate: (r: Route) => void }) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 4);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={cn(
        "sticky top-0 z-30 w-full border-b transition-colors duration-300",
        scrolled ? "border-white/10 bg-[var(--bg-0)]/85 backdrop-blur-md" : "border-transparent",
      )}
    >
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4 sm:px-10 lg:px-16">
        <button onClick={() => onNavigate("home")} className="group flex items-center gap-3">
          <span className="grid h-8 w-8 place-items-center rounded-md border border-white/15 bg-white/[0.03] text-[var(--accent)]">
            <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
              <path d="M3 17V7a1 1 0 011-1h2v11M9 17V3a1 1 0 011-1h2v15M15 17v-7a1 1 0 011-1h2v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-[var(--ink-1)]">
            DataLens AI
          </span>
          <span className="hidden font-mono text-[10px] uppercase tracking-[0.32em] text-[var(--ink-4)] sm:inline">
            v3
          </span>
        </button>

        <nav className="flex items-center gap-1">
          {ROUTES.map((r) => (
            <NavLink key={r.id} active={route === r.id} onClick={() => onNavigate(r.id)} label={r.label} />
          ))}
        </nav>
      </div>
    </header>
  );
}

function NavLink({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative rounded-md px-3 py-1.5 text-sm transition-colors duration-200",
        active ? "text-[var(--ink-1)]" : "text-[var(--ink-3)] hover:text-[var(--ink-1)]",
      )}
    >
      {label}
      {active ? (
        <motion.span
          layoutId="nav-underline"
          className="absolute inset-x-2 -bottom-px h-px bg-[var(--accent)]"
          transition={{ type: "spring", stiffness: 500, damping: 32 }}
        />
      ) : null}
    </button>
  );
}

/* ----------------------------- Footer ------------------------------------ */

function Footer() {
  return (
    <footer className="mx-auto w-full max-w-6xl border-t border-[var(--line)] px-6 py-12 sm:px-10 lg:px-16">
      <div className="hr-rule mb-6">FIN</div>
      <div className="flex flex-col gap-4 text-[12px] text-[var(--ink-3)] sm:flex-row sm:items-center sm:justify-between">
        <span className="font-mono uppercase tracking-[0.32em]">
          DataLens AI · v3 · LLM analytics
        </span>
        <div className="flex flex-wrap gap-x-6 gap-y-1">
          <span>Flask · 5055</span>
          <span>Vite · 5173</span>
          <span>Ollama · 11434</span>
        </div>
      </div>
    </footer>
  );
}

/* ----------------------------- Primitives -------------------------------- */

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn("eyebrow", className)}>{children}</p>;
}

export function PageTitle({
  children,
  subtitle,
}: {
  children: ReactNode;
  subtitle?: ReactNode;
}) {
  return (
    <div className="mb-12 max-w-3xl">
      <motion.h1
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="display-1"
      >
        {children}
      </motion.h1>
      {subtitle ? (
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.06, ease: "easeOut" }}
          className="lede mt-6"
        >
          {subtitle}
        </motion.p>
      ) : null}
    </div>
  );
}

export function SectionHeader({
  eyebrow,
  title,
  description,
  className,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mb-8 max-w-3xl", className)}>
      {eyebrow ? <Eyebrow className="mb-3">{eyebrow}</Eyebrow> : null}
      <h2 className="display-2">{title}</h2>
      {description ? <p className="lede mt-3">{description}</p> : null}
    </div>
  );
}

export function Section({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("py-12 sm:py-16", className)}>{children}</section>
  );
}

export function Hr({ label }: { label?: string }) {
  return (
    <div className="my-12 sm:my-16">
      {label ? <div className="hr-rule">{label}</div> : <hr className="border-[var(--line)]" />}
    </div>
  );
}

/* --------------------------- ErrorBoundary ------------------------------- */
/**
 * Catches render-time errors anywhere in the page tree so the user never
 * sees a white screen. Useful when a payload field comes back in an
 * unexpected shape.
 */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: { componentStack?: string }) {
    // eslint-disable-next-line no-console
    console.error("DataLens render error:", error, info);
  }
  render() {
    if (this.state.error) {
      return (
        <main className="mx-auto w-full max-w-3xl px-6 py-24 sm:px-10 lg:px-16">
          <p className="eyebrow mb-3">Something broke</p>
          <h2 className="display-2">A panel crashed while rendering.</h2>
          <p className="lede mt-4">
            The data came back from the backend but one of the panels couldn't render it. The
            dataset itself is fine.
          </p>
          <pre className="mt-8 overflow-x-auto rounded-xl border border-[var(--line)] bg-[var(--bg-1)] p-4 text-xs text-[var(--ink-2)]">
{this.state.error.message}
          </pre>
          <button
            onClick={() => {
              this.setState({ error: null });
              window.location.hash = "#/home";
            }}
            className="btn-ghost mt-6"
          >
            ← Back to overview
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}
