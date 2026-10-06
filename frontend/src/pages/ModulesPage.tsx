import { Eyebrow, PageTitle, Section } from "@/components/site/SiteShell";

const PROTOCOLS = [
  {
    name: "Prism",
    role: "Understand",
    body: "Build one coherent view of a dataset and surface the signals that deserve attention.",
    state: "Primary",
  },
  {
    name: "Axiom",
    role: "Trust",
    body: "Establish expectations from trusted data and test whether new data still respects them.",
    state: "Available",
  },
  {
    name: "Forge",
    role: "Transform",
    body: "Prepare a conservative transformation plan and apply it only when you choose to.",
    state: "Available",
  },
  {
    name: "Tide",
    role: "Compare",
    body: "Read how a dataset changes between two points in time, releases, or pipeline states.",
    state: "Available",
  },
  {
    name: "Pulse",
    role: "Monitor",
    body: "Capture compact health states so the evolution of a dataset can be followed over time.",
    state: "Available",
  },
] as const;

export function ModulesPage() {
  return (
    <>
      <Section className="pb-8 pt-6">
        <Eyebrow>Protocol system</Eyebrow>
        <PageTitle subtitle="FrameVitals exposes a small set of named workflows instead of making you assemble the analysis stack yourself.">
          Choose the intent, not the implementation.
        </PageTitle>
      </Section>

      <Section className="py-0">
        <div className="grid gap-4 sm:gap-6 md:grid-cols-2">
          {PROTOCOLS.map((protocol) => (
            <article
              key={protocol.name}
              className="card group flex min-h-52 flex-col justify-between p-6 transition-transform duration-300 hover:-translate-y-0.5"
            >
              <div>
                <p className="eyebrow">{protocol.role}</p>
                <h3 className="mt-2 text-2xl font-semibold text-[var(--ink-1)]">
                  {protocol.name}
                </h3>
                <p className="mt-3 text-sm leading-7 text-[var(--ink-2)]">
                  {protocol.body}
                </p>
              </div>
              <p className="mt-8 font-mono text-[10px] uppercase tracking-[0.24em] text-[var(--ink-4)]">
                {protocol.state}
              </p>
            </article>
          ))}
        </div>

        <div className="mt-12 rounded-2xl border border-[var(--line)] bg-[var(--bg-1)] p-6">
          <Eyebrow>For developers</Eyebrow>
          <p className="mt-3 max-w-3xl text-sm leading-7 text-[var(--ink-2)]">
            The technical composition of every protocol remains documented and inspectable.
            The product interface stays intentionally high-level so protocol names describe
            intent rather than exposing the machinery behind them.
          </p>
        </div>
      </Section>
    </>
  );
}
