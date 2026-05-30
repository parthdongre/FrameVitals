import { useEffect, useState } from "react";

export interface NavSection {
  id: string;
  label: string;
  hint?: string;
}

interface DashboardNavProps {
  sections: NavSection[];
}

/**
 * Sticky horizontal pill nav that scrolls to an anchored section and
 * highlights whichever section is currently in view.
 */
export function DashboardNav({ sections }: DashboardNavProps) {
  const [activeId, setActiveId] = useState<string | null>(sections[0]?.id ?? null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]) {
          setActiveId(visible[0].target.id);
        }
      },
      {
        rootMargin: "-30% 0px -55% 0px",
        threshold: [0, 0.25, 0.5, 0.75, 1],
      },
    );

    sections.forEach((s) => {
      const el = document.getElementById(s.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [sections]);

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - 80;
    window.scrollTo({ top, behavior: "smooth" });
    setActiveId(id);
  };

  return (
    <div className="sticky top-0 z-30 -mx-4 mb-2 border-b border-white/5 bg-space-950/85 px-4 py-2 backdrop-blur-md sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8 xl:-mx-10 xl:px-10">
      <div className="flex items-center gap-2 overflow-x-auto pb-1">
        {sections.map((s) => {
          const active = s.id === activeId;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => scrollTo(s.id)}
              className={`whitespace-nowrap rounded-full border px-3 py-1 text-xs font-semibold transition ${
                active
                  ? "border-cyan-400/40 bg-cyan-500/15 text-cyan-100 shadow-[0_0_18px_rgba(6,182,212,0.18)]"
                  : "border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/20 hover:text-slate-200"
              }`}
              title={s.hint}
            >
              {s.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
