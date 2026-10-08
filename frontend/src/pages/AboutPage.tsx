import { Eyebrow, Hr, PageTitle, Section, SectionHeader } from "@/components/site/SiteShell";

export function AboutPage() {
  return (
    <>
      <Section className="pb-12 pt-6">
        <Eyebrow>About</Eyebrow>
        <PageTitle subtitle="FrameVitals is an open-source Python diagnostics framework for understanding, trusting, comparing, and monitoring structured data and ML systems through a small protocol-first interface.">
          Diagnostics without the assembly work.
        </PageTitle>
      </Section>

      <Hr label="Design" />

      <Section>
        <SectionHeader
          eyebrow="Protocol-first"
          title="Choose the intent. FrameVitals handles the workflow."
          description="The public experience is organized around Prism, Axiom, Forge, Tide, and Pulse. The implementation remains inspectable in the documentation and source code without turning the product surface into a list of internal stages."
        />

        <div className="grid gap-x-12 gap-y-8 md:grid-cols-2">
          <Stat label="Prism" value="Understand a structured source or model as one coherent report." />
          <Stat label="Axiom" value="Establish and test expectations." />
          <Stat label="Forge" value="Prepare and apply careful transformations." />
          <Stat label="Tide" value="Read meaningful change between data, graph, tensor, or model states." />
          <Stat label="Pulse" value="Capture compact data and model health states over time." />
          <Stat label="Principle" value="Evidence first, protocol surface second, internals documented." />
        </div>
      </Section>

      <Hr label="Boundaries" />

      <Section>
        <SectionHeader eyebrow="What it is" title="A structured, reproducible diagnostic layer." />
        <ul className="space-y-3 text-[14px] leading-7 text-[var(--ink-2)]">
          <li>· Designed to turn tables, graphs, tensors, nested structures, relational projects, and ML artifacts into coherent findings and next actions.</li>
          <li>· Built so protocol results can be inspected, exported, and reproduced.</li>
          <li>· Intended to complement domain expertise rather than replace it.</li>
          <li>· Optional interpretation features remain secondary to computed evidence.</li>
        </ul>
      </Section>

      <Section>
        <SectionHeader eyebrow="What it isn't" title="Not a black box." />
        <ul className="space-y-3 text-[14px] leading-7 text-[var(--ink-2)]">
          <li>· Protocol names simplify the user experience; they do not hide the implementation from developers.</li>
          <li>· Findings are signals to investigate, not automatic proof that data is wrong.</li>
          <li>· Model-oriented outputs are analytical baselines, not production sign-off.</li>
        </ul>
      </Section>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="eyebrow">{label}</p>
      <p className="mt-2 text-base text-[var(--ink-1)]">{value}</p>
    </div>
  );
}
