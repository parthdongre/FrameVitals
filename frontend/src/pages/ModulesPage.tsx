import { useState } from "react";
import { Eyebrow, PageTitle, Section } from "@/components/site/SiteShell";
import { MODULES, type ModuleRegistryEntry } from "@/data/moduleRegistry";
import { cn } from "@/lib/utils";

const CATEGORIES: { id: ModuleRegistryEntry["category"] | "All"; label: string }[] = [
  { id: "All", label: "All" },
  { id: "Profiling", label: "Profiling" },
  { id: "Quality", label: "Quality" },
  { id: "Statistics", label: "Statistics" },
  { id: "ML", label: "ML" },
  { id: "Time", label: "Time-series" },
  { id: "Text", label: "Text" },
  { id: "Drift", label: "Drift" },
  { id: "Cleaning", label: "Cleaning" },
  { id: "AI", label: "AI" },
];

export function ModulesPage() {
  const [filter, setFilter] = useState<ModuleRegistryEntry["category"] | "All">("All");
  const filtered = filter === "All" ? MODULES : MODULES.filter((module) => module.category === filter);

  return (
    <>
      <Section className="pb-8 pt-6">
        <Eyebrow>Module reference</Eyebrow>
        <PageTitle
          subtitle={`FrameVitals currently exposes ${MODULES.length} analytical components. This reference explains what each component produces, how it works, and where its canonical implementation lives.`}
        >
          What's under the hood.
        </PageTitle>
      </Section>

      <Section className="py-0">
        <div className="-mx-6 mb-12 flex gap-2 overflow-x-auto px-6 sm:mx-0 sm:px-0">
          {CATEGORIES.map((category) => {
            const active = filter === category.id;
            const count =
              category.id === "All"
                ? MODULES.length
                : MODULES.filter((module) => module.category === category.id).length;

            return (
              <button
                key={category.id}
                onClick={() => setFilter(category.id)}
                className={cn(
                  "shrink-0 rounded-full border px-4 py-1.5 text-xs font-medium transition-colors duration-200",
                  active
                    ? "border-[var(--accent-line)] bg-[var(--accent-soft)] text-[var(--ink-1)]"
                    : "border-[var(--line)] text-[var(--ink-3)] hover:border-[var(--line-strong)] hover:text-[var(--ink-1)]",
                )}
              >
                {category.label}
                <span className="ml-2 text-[10px] text-[var(--ink-4)]">{count}</span>
              </button>
            );
          })}
        </div>

        <div className="grid gap-4 sm:gap-6 md:grid-cols-2">
          {filtered.map((module) => (
            <ModuleCard key={module.id} module={module} />
          ))}
        </div>
      </Section>
    </>
  );
}

function ModuleCard({ module }: { module: ModuleRegistryEntry }) {
  const [open, setOpen] = useState(false);

  return (
    <article className="card group flex flex-col p-6 transition-transform duration-300 hover:-translate-y-0.5">
      <p className="eyebrow">{module.category}</p>
      <h3 className="mt-2 text-xl font-semibold text-[var(--ink-1)]">{module.name}</h3>
      <p className="mt-2 text-sm leading-7 text-[var(--ink-2)]">{module.oneLiner}</p>

      <button
        onClick={() => setOpen((value) => !value)}
        className="mt-5 inline-flex w-fit items-center gap-2 text-xs font-medium text-[var(--accent)] hover:text-[var(--ink-1)]"
      >
        {open ? "Hide details" : "Read details"}
        <span className={cn("transition", open ? "rotate-90" : "")}>→</span>
      </button>

      {open ? (
        <div className="mt-5 space-y-5 border-t border-[var(--line)] pt-5 text-sm leading-7 text-[var(--ink-2)]">
          <DetailBlock label="What it does" body={module.what} />
          <DetailBlock label="How it works" body={module.how} />

          <div>
            <Eyebrow className="mb-2">Algorithms</Eyebrow>
            <div className="flex flex-wrap gap-1.5">
              {module.algorithms.map((algorithm) => (
                <span
                  key={algorithm}
                  className="rounded-full border border-[var(--line)] bg-[var(--bg-2)] px-2 py-0.5 text-[11px] text-[var(--ink-2)]"
                >
                  {algorithm}
                </span>
              ))}
            </div>
          </div>

          <div>
            <Eyebrow className="mb-2">Source</Eyebrow>
            <p className="font-mono text-[11px] text-[var(--ink-3)]">{module.source}</p>
          </div>
        </div>
      ) : null}
    </article>
  );
}

function DetailBlock({ label, body }: { label: string; body: string }) {
  return (
    <div>
      <Eyebrow className="mb-2">{label}</Eyebrow>
      <p className="text-pretty text-sm leading-7 text-[var(--ink-2)]">{body}</p>
    </div>
  );
}
