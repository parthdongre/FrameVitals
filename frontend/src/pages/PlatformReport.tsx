import { useState } from "react";
import { ArrowUpRight, Download } from "lucide-react";
import {
  Eyebrow,
  PageTitle,
  Section,
  type Route,
} from "@/components/site/SiteShell";
import { asObject, display, label, type PlatformResult } from "@/lib/platform";

function Metric({ name, value }: { name: string; value: unknown }) {
  return (
    <div className="metric-card">
      <span className="workspace-muted capitalize">{name}</span>
      <strong>{display(value)}</strong>
    </div>
  );
}
function Detail({
  name,
  value,
  level = 0,
}: {
  name: string;
  value: unknown;
  level?: number;
}) {
  const [open, setOpen] = useState(false);
  if (value === null || typeof value !== "object")
    return (
      <dl className="result-dl">
        <dt>{label(name)}</dt>
        <dd>{display(value)}</dd>
      </dl>
    );
  const entries = Object.entries(value);
  return (
    <details
      className="result-details"
      onToggle={(e) => setOpen(e.currentTarget.open)}
    >
      <summary>
        {label(name)}{" "}
        <span className="workspace-muted">
          {" "}
          · {entries.length} {Array.isArray(value) ? "items" : "fields"}
        </span>
      </summary>
      {open &&
        (level < 6 ? (
          entries
            .slice(0, 40)
            .map(([key, item]) => (
              <Detail key={key} name={key} value={item} level={level + 1} />
            ))
        ) : (
          <p className="workspace-muted mt-3">
            Download JSON to inspect deeper structures.
          </p>
        ))}
      {open && entries.length > 40 && (
        <p className="workspace-muted mt-4">
          Showing 40 of {entries.length}. The JSON export contains the full
          result.
        </p>
      )}
    </details>
  );
}
function Bars({
  title,
  values,
  percent = false,
}: {
  title: string;
  values: Record<string, unknown>;
  percent?: boolean;
}) {
  const entries = Object.entries(values)
    .filter(
      (entry): entry is [string, number] =>
        typeof entry[1] === "number" && Number.isFinite(entry[1]),
    )
    .slice(0, 20);
  if (!entries.length) return null;
  const max = percent
    ? 100
    : Math.max(...entries.map(([, v]) => Math.abs(v)), 1);
  return (
    <section className="workspace-panel">
      <h2 className="mb-5 text-lg">{title}</h2>
      <div className="space-y-4">
        {entries.map(([name, value]) => (
          <div key={name}>
            <div className="mb-2 flex justify-between gap-4 text-xs">
              <span className="truncate">{label(name)}</span>
              <span>
                {display(value)}
                {percent ? "%" : ""}
              </span>
            </div>
            <div className="h-2 rounded bg-white/10">
              <div
                style={{
                  width: `${Math.min(100, (Math.abs(value) / max) * 100)}%`,
                }}
                className="h-2 rounded bg-[var(--accent)]"
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
function NetworkExcerpt({ value }: { value: unknown }) {
  const data = asObject(value);
  const nodes = data.nodes as { id: number; label: string }[] | undefined;
  const edges = data.edges as [number, number][] | undefined;
  if (!nodes?.length || !edges) return null;
  const points = nodes.map((node, i) => ({
    ...node,
    x: 250 + 132 * Math.cos((i / nodes.length) * Math.PI * 2),
    y: 170 + 132 * Math.sin((i / nodes.length) * Math.PI * 2),
  }));
  return (
    <section className="workspace-panel">
      <h2 className="text-lg">Network excerpt</h2>
      <p className="workspace-muted mt-2">
        {nodes.length} actual nodes · {edges.length} edges · circular display
        layout{data.sampled ? " · partial view" : ""}
      </p>
      <svg
        viewBox="0 0 500 340"
        role="img"
        aria-label={`Graph excerpt with ${nodes.length} nodes and ${edges.length} edges`}
        className="mt-3 w-full"
      >
        <defs>
          <marker
            id="edge-arrow"
            markerWidth="6"
            markerHeight="6"
            refX="12"
            refY="3"
            orient="auto"
          >
            <path d="M0 0 L6 3 L0 6" fill="#7aa990" />
          </marker>
        </defs>
        {edges.map(([a, b], i) =>
          points[a] && points[b] ? (
            <line
              key={i}
              x1={points[a].x}
              y1={points[a].y}
              x2={points[b].x}
              y2={points[b].y}
              stroke="#7aa990"
              strokeOpacity="0.5"
              markerEnd={data.directed ? "url(#edge-arrow)" : undefined}
            />
          ) : null,
        )}
        {points.map((p) => (
          <g key={p.id}>
            <circle
              cx={p.x}
              cy={p.y}
              r={nodes.length > 25 ? 4 : 7}
              fill="#5eead4"
            >
              <title>{p.label}</title>
            </circle>
            {nodes.length <= 15 && (
              <text
                x={p.x}
                y={p.y - 13}
                fill="#dbecdf"
                fontSize="12"
                textAnchor="middle"
              >
                {p.label.slice(0, 18)}
              </text>
            )}
          </g>
        ))}
      </svg>
    </section>
  );
}
export function PlatformReport({
  result,
  onNavigate,
}: {
  result: PlatformResult;
  onNavigate: (route: Route) => void;
}) {
  const a = result.analysis,
    kind = result.source.kind;
  const profile = asObject(a.profile),
    shape = asObject(profile.shape),
    domain = asObject(a[kind]);
  const health = asObject(a.health);
  const metrics: [string, unknown][] =
    kind === "graph"
      ? [
          ["Nodes", domain.nodes],
          ["Edges", domain.edges],
          ["Density", domain.density],
          ["Isolated nodes", domain.isolates],
        ]
      : kind === "tensor"
        ? [
            ["Shape", domain.shape],
            ["Dtype", domain.dtype],
            ["Values", domain.size],
            [
              "Sample coverage",
              typeof domain.sample_fraction === "number"
                ? `${(domain.sample_fraction * 100).toFixed(1)}%`
                : undefined,
            ],
          ]
        : kind === "document"
          ? [
              ["Words extracted", domain.words],
              ["Characters", domain.characters],
              [
                "Pages analyzed",
                domain.pages_analyzed ?? domain.slides_analyzed,
              ],
              ["Total pages", domain.pages ?? domain.slides],
            ]
          : kind === "model"
            ? [
                ["Parameters", domain.total_parameters ?? domain.parameters],
                ["Framework", domain.framework],
                ["Tensors", domain.tensor_count ?? domain.parameter_tensors],
                ["Modules", domain.modules],
              ]
            : kind === "relational"
              ? [
                  ["Tables", domain.table_count],
                  ["Relationships", domain.relationship_count],
                  ["Rows", shape.rows],
                ]
              : kind === "nested"
                ? [
                    ["Nodes inspected", domain.nodes_observed],
                    ["Maximum depth", domain.max_depth],
                    ["Null values", domain.null_values],
                    ["Type conflicts", domain.path_type_conflict_count],
                  ]
                : [
                    ["Rows", shape.rows],
                    ["Columns", shape.columns],
                    [
                      "Missing cells",
                      asObject(health.details).missing_percent == null
                        ? null
                        : `${display(asObject(health.details).missing_percent)}%`,
                    ],
                    ["Duplicate rows", profile.duplicate_rows],
                  ];
  function download() {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "framevitals-result.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const sections =
    kind === "tabular"
      ? [
          "profile",
          "quality_diagnostics",
          "advanced",
          "deep_statistics_v2",
          "anomalies_v2",
          "ml_readiness",
          "target_intelligence",
          "model_leaderboard",
          "explainability",
          "time_series",
          "text_profile",
        ]
      : [kind];
  return (
    <>
      <Section className="workspace-heading pb-8 pt-6">
        <Eyebrow>Prism / {kind} result</Eyebrow>
        <PageTitle subtitle={result.filename}>
          Your source, understood.
        </PageTitle>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <span className="workspace-badge">{result.status}</span>
          <span className="workspace-muted">
            {result.elapsed_ms == null
              ? "Timing unavailable"
              : `${(result.elapsed_ms / 1000).toFixed(2)} s`}{" "}
            · {String(a.analysis_mode)} depth · coverage{" "}
            {display(result.execution.coverage)}
          </span>
        </div>
        <div className="mt-6 flex flex-wrap gap-3">
          <button className="btn-primary" onClick={download}>
            <Download size={16} /> Export JSON
          </button>
          <button className="btn-ghost" onClick={() => onNavigate("analyze")}>
            New analysis <ArrowUpRight size={16} />
          </button>
        </div>
      </Section>
      <div className="metric-grid mb-6">
        {metrics
          .filter(([, v]) => v !== undefined)
          .map(([name, value]) => (
            <Metric key={name} name={name} value={value} />
          ))}
      </div>
      <div className="workspace-grid">
        <div className="space-y-5">
          {kind === "tabular" && (
            <Bars
              title="Missing values by column"
              values={asObject(profile.missing_percent)}
              percent
            />
          )}
          {kind === "tensor" && (
            <Bars
              title="Value diagnostics"
              values={Object.fromEntries(
                Object.entries(domain)
                  .filter(([k]) =>
                    [
                      "zero_fraction_sample",
                      "nan_fraction_sample",
                      "inf_fraction_sample",
                      "near_zero_fraction_sample",
                    ].includes(k),
                  )
                  .map(([k, v]) => [k, typeof v === "number" ? v * 100 : null]),
              )}
              percent
            />
          )}
          {kind === "graph" && (
            <NetworkExcerpt value={domain.topology_preview} />
          )}
          {kind === "tensor" && (
            <div>
              <Bars
                title="Singular values (up to 24)"
                values={Object.fromEntries(
                  ((asObject(domain.matrix).singular_values ?? []) as number[])
                    .slice(0, 24)
                    .map((v, i) => [String(i + 1), v]),
                )}
              />
              <p className="workspace-muted mt-2">
                {!asObject(domain.matrix).method
                  ? "No valid matrix spectrum is available. "
                  : asObject(domain.matrix).population === "sampled_submatrix"
                  ? "Spectrum of a sampled submatrix; it does not estimate full-matrix rank. "
                  : "Spectrum of the full matrix. "}
                {!asObject(domain.matrix).method
                  ? ""
                  : asObject(domain.matrix).approximate
                  ? "Randomized approximation; displayed values may be truncated."
                  : "Full SVD; displayed values may be truncated."}
              </p>
            </div>
          )}
          {kind === "graph" && (
            <Bars title="Degree summary" values={asObject(domain.degree)} />
          )}
          {kind === "nested" && (
            <Bars title="Value types" values={asObject(domain.type_counts)} />
          )}
          {kind === "model" && (
            <Bars
              title="Operator counts"
              values={asObject(domain.operator_counts)}
            />
          )}
          <section className="workspace-panel">
            <h2 className="text-lg">Findings & next steps</h2>
            <p className="workspace-muted mt-2">
              Findings reflect the diagnostics that ran, including any sampling
              or approximation.
            </p>
            {result.findings.length ? (
              result.findings.slice(0, 50).map((f, i) => (
                <article className="finding-card" key={f.id ?? i}>
                  <span className="workspace-muted uppercase">
                    {f.severity ?? "info"}
                  </span>
                  <h3 className="mt-1 text-sm font-semibold">
                    {f.title ?? f.name ?? f.id ?? "Diagnostic finding"}
                  </h3>
                  <p className="workspace-muted mt-2">{f.summary}</p>
                  {f.recommendation && (
                    <p className="mt-3 text-sm text-emerald-200">
                      {f.recommendation}
                    </p>
                  )}
                </article>
              ))
            ) : (
              <p className="mt-5 text-sm">
                No findings were returned by the completed diagnostics. Review
                coverage before drawing conclusions.
              </p>
            )}
          </section>
          {result.change && (
            <section className="workspace-panel">
              <h2 className="text-lg">Reference comparison</h2>
              <Detail name="Change results" value={result.change} />
            </section>
          )}
          <section className="workspace-panel">
            <h2 className="text-lg">Explore the evidence</h2>
            <p className="workspace-muted mt-2">
              Expand a section for measured values and method details. Large
              structures are bounded in this view.
            </p>
            {sections
              .filter((key) => key in a)
              .map((key) => (
                <Detail key={key} name={key} value={a[key]} />
              ))}
          </section>
        </div>
        <aside className="space-y-5">
          <section className="workspace-panel">
            <p className="label-mono">Diagnostic score</p>
            <p className="mt-3 text-5xl font-semibold text-[var(--accent)]">
              {display(health.overall_score)}
              <span className="ml-2 text-sm text-[var(--ink-2)]">/ 100</span>
            </p>
            <p className="mt-3 text-sm">{display(health.label)}</p>
            <p className="workspace-muted mt-4">
              {result.score_interpretation}
            </p>
          </section>
          <section className="workspace-panel">
            <h2 className="text-lg">Execution & coverage</h2>
            <p className="workspace-muted mt-2">
              Unavailable means unmeasured. Approximate and sampled results
              retain their method labels.
            </p>
            <div className="mt-5 space-y-4">
              {result.diagnostics.map((d) => (
                <div key={d.name} className="border-b border-white/10 pb-3">
                  <div className="flex flex-wrap justify-between gap-2 text-xs">
                    <strong className="capitalize">{label(d.name)}</strong>
                    <span className="text-[var(--accent)]">
                      {label(d.status)}
                    </span>
                  </div>
                  <p className="workspace-muted mt-1">
                    {d.method ? `${label(d.method)} · ` : ""}
                    {d.coverage} coverage · {d.accuracy} accuracy
                  </p>
                  {d.reason && (
                    <p className="workspace-muted">{label(d.reason)}</p>
                  )}
                </div>
              ))}
            </div>
            <Detail name="Execution metadata" value={result.execution} />
          </section>
        </aside>
      </div>
    </>
  );
}
