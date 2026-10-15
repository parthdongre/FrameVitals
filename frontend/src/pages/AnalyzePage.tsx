import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  Check,
  FileUp,
  Layers3,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import {
  Eyebrow,
  PageTitle,
  Section,
  type Route,
} from "@/components/site/SiteShell";
import { getJSON, postFormData } from "@/lib/api";
import {
  setPlatformResult,
  type PlatformResult,
  type SourcePreview,
} from "@/lib/platform";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

interface Catalog {
  formats: {
    extensions: string[];
    installed: boolean;
    label: string;
    extra: string | null;
  }[];
  upload_limit_bytes: number;
}
const MODES = [
  ["quick", "Quick", "A bounded first look."],
  ["standard", "Standard", "Balanced coverage and speed."],
  ["deep", "Deep", "Larger analysis budgets."],
  ["research", "Research", "The largest built-in budgets."],
];
export function AnalyzePage({
  onNavigate,
}: {
  onNavigate: (route: Route) => void;
}) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [reference, setReference] = useState<File | null>(null);
  const [preview, setPreview] = useState<SourcePreview | null>(null);
  const [mode, setMode] = useState("standard");
  const [target, setTarget] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [running, setRunning] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const sequence = useRef(0);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    const c = new AbortController();
    getJSON<Catalog>("/api/capabilities", { signal: c.signal })
      .then(setCatalog)
      .catch((e) => {
        if (!c.signal.aborted) setError(e.message);
      });
    return () => {
      c.abort();
      controller.current?.abort();
    };
  }, []);
  async function choose(next: File) {
    const n = ++sequence.current;
    controller.current?.abort();
    const c = new AbortController();
    controller.current = c;
    setFile(next);
    setPreview(null);
    setError(null);
    setTarget("");
    setReference(null);
    if (next.size > (catalog?.upload_limit_bytes ?? 50 * 1024 * 1024)) {
      setError("This file exceeds the 50 MiB upload limit.");
      setInspecting(false);
      return;
    }
    setInspecting(true);
    try {
      const fd = new FormData();
      fd.append("dataset", next);
      const result = await postFormData<SourcePreview>("/api/inspect", fd, {
        signal: c.signal,
        timeoutMs: 60000,
      });
      if (sequence.current === n) setPreview(result);
    } catch (e) {
      if (sequence.current === n && !c.signal.aborted)
        setError((e as Error).message);
    } finally {
      if (sequence.current === n) setInspecting(false);
    }
  }
  async function run() {
    if (!file || !preview || running) return;
    setRunning(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("dataset", file);
      fd.append("analysis_mode", mode);
      if (target) fd.append("target_column", target);
      if (reference) fd.append("reference", reference);
      const result = await postFormData<PlatformResult>("/api/prism", fd, {
        timeoutMs: 300000,
      });
      if (result.schema_version !== "1" || !result.analysis || !result.id)
        throw new Error("The server returned an unsupported result contract.");
      setPlatformResult(result);
      onNavigate("report");
    } catch (e) {
      setError((e as Error).message || "Analysis failed.");
    } finally {
      setRunning(false);
    }
  }
  return (
    <>
      <Section className="workspace-heading pb-8 pt-6">
        <Eyebrow>Analysis workspace / Prism</Eyebrow>
        <PageTitle subtitle="Bring your data. See what it can tell you.">
          One source. A clearer picture.
        </PageTitle>
      </Section>
      <div className="workspace-grid">
        <div className="space-y-5">
          <section className="workspace-panel" aria-labelledby="source-heading">
            <div className="flex items-center justify-between">
              <h2 id="source-heading" className="text-lg font-semibold">
                01 <span className="ml-3">Choose a source</span>
              </h2>
              <span className="workspace-muted">50 MiB max</span>
            </div>
            <button
              type="button"
              disabled={running}
              className={cn("source-drop", dragging && "source-drop-active")}
              onClick={() => input.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                if (!running && e.dataTransfer.files[0])
                  void choose(e.dataTransfer.files[0]);
              }}
            >
              <span className="source-icon">
                <FileUp size={26} aria-hidden="true" />
              </span>
              <strong className="break-all text-lg">
                {file?.name ?? "Drop a file to begin"}
              </strong>
              <span className="workspace-muted">
                {file
                  ? `${formatBytes(file.size)} · Click to replace`
                  : "or browse your files"}
              </span>
              <span className="workspace-muted text-xs">
                Tables · graphs · tensors · documents · models
              </span>
            </button>
            <input
              ref={input}
              aria-label="Source file"
              type="file"
              className="sr-only"
              tabIndex={-1}
              accept={catalog?.formats.flatMap((f) => f.extensions).join(",")}
              onChange={(e) => {
                if (e.target.files?.[0]) void choose(e.target.files[0]);
                e.target.value = "";
              }}
            />
            <p className="mt-4 flex gap-2 text-xs text-[var(--ink-2)]">
              <ShieldCheck size={16} className="shrink-0" aria-hidden="true" />
              Preview and analysis upload this file to your configured server.
              This workflow does not use external AI. Raw uploads are removed
              after each request.
            </p>
            <div aria-live="polite" className="mt-5">
              {inspecting && (
                <p className="flex items-center gap-2">
                  <Loader2 size={16} className="animate-spin" />
                  Inspecting source…
                </p>
              )}
              {preview && (
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="workspace-badge">
                      <Check size={12} /> {preview.source.kind} detected
                    </span>
                    <span className="workspace-muted">Ready to configure</span>
                  </div>
                  {preview.preview?.columns && (
                    <div className="preview-scroll mt-4">
                      <table className="workspace-table">
                        <caption className="sr-only">
                          First six rows, up to forty columns
                        </caption>
                        <thead>
                          <tr>
                            {preview.preview.columns.map((c) => (
                              <th key={c} scope="col">
                                {c}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {preview.preview.rows?.map((row, i) => (
                            <tr key={i}>
                              {row.map((v, j) => (
                                <td key={j}>
                                  {v == null ? (
                                    <span className="text-amber-200">
                                      missing
                                    </span>
                                  ) : (
                                    String(v)
                                  )}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {preview.preview?.shape && (
                    <p className="workspace-muted mt-3">
                      Shape {preview.preview.shape.join(" × ")} ·{" "}
                      {preview.preview.dtype}
                    </p>
                  )}
                </>
              )}
            </div>
          </section>
          <section className="workspace-panel" aria-labelledby="config-heading">
            <h2 id="config-heading" className="mb-5 text-lg font-semibold">
              02 <span className="ml-3">Set the depth</span>
            </h2>
            <fieldset disabled={running}>
              <legend className="sr-only">Analysis depth</legend>
              <div className="grid grid-cols-2 gap-2">
                {MODES.map(([id, title, desc]) => (
                  <label
                    key={id}
                    className={cn(
                      "mode-choice",
                      mode === id && "mode-selected",
                    )}
                  >
                    <input
                      type="radio"
                      name="depth"
                      value={id}
                      checked={mode === id}
                      onChange={() => setMode(id)}
                    />
                    <span>
                      <strong>{title}</strong>
                      <small>{desc}</small>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
            {preview?.source.supports_focus && (
              <label className="mt-5 block text-sm">
                Focus column <span className="workspace-muted">(optional)</span>
                <input
                  aria-label="Focus column"
                  list="source-columns"
                  className="workspace-input mt-2"
                  value={target}
                  disabled={running}
                  onChange={(e) => setTarget(e.target.value)}
                  placeholder="Choose an outcome column"
                />
                <datalist id="source-columns">
                  {preview.preview?.columns?.map((c) => (
                    <option key={c} value={c} />
                  ))}
                </datalist>
              </label>
            )}
            {preview?.source.supports_reference && (
              <details className="mt-5">
                <summary className="cursor-pointer text-sm text-[var(--ink-2)]">
                  Compare with a reference (optional)
                </summary>
                <label className="mt-3 block text-sm">
                  Reference file
                  <input
                    aria-label="Reference file"
                    className="workspace-input mt-2 py-2"
                    type="file"
                    disabled={running}
                    onChange={(e) => setReference(e.target.files?.[0] ?? null)}
                  />
                </label>
                <p className="workspace-muted mt-2">
                  Use the same source kind. Both files share the 50 MiB request
                  limit.
                </p>
              </details>
            )}
          </section>
          {error && (
            <div role="alert" className="workspace-error">
              <strong>Couldn’t complete this step</strong>
              <p className="mt-1 break-words text-sm">{error}</p>
            </div>
          )}
          <button
            className="btn-primary w-full justify-center disabled:opacity-40"
            disabled={!preview || inspecting || running}
            onClick={() => void run()}
          >
            {running ? (
              <>
                <Loader2 size={18} className="animate-spin" /> Running Prism…
              </>
            ) : (
              <>
                Run analysis <ArrowUpRight size={18} />
              </>
            )}
          </button>
          {running && (
            <p role="status" className="workspace-muted">
              Analysis is running on the server. Larger budgets can take several
              minutes.
            </p>
          )}
        </div>
        <aside className="space-y-5">
          <section className="workspace-panel">
            <div className="mb-5 flex items-center gap-2 text-[var(--accent)]">
              <Layers3 size={18} />
              <span className="label-mono">Available analyses</span>
            </div>
            <h2 className="mb-4 text-xl font-semibold">
              {preview
                ? `Made for ${preview.source.kind} sources.`
                : "A workflow that fits your source."}
            </h2>
            <p className="workspace-muted mb-5">
              {preview
                ? "These analyses apply to your source. Individual diagnostics may be skipped when data, dependencies, or budgets limit execution."
                : "Select a file to detect its format and discover the analyses available on this server."}
            </p>
            <ul className="space-y-3">
              {preview?.source.analyses.map((name) => (
                <li key={name} className="flex gap-3 text-sm">
                  <Check size={16} className="shrink-0 text-[var(--accent)]" />
                  {name}
                </li>
              ))}
            </ul>
            {preview?.source.limitations.map((l) => (
              <p key={l} className="mt-4 text-xs text-amber-200">
                {l}
              </p>
            ))}
          </section>
          <details className="workspace-panel">
            <summary className="cursor-pointer font-semibold">
              Supported formats
            </summary>
            <ul className="mt-4 space-y-3 text-xs">
              {catalog?.formats.map((f) => (
                <li key={f.label}>
                  <strong>{f.extensions.join(" · ")}</strong>
                  <p className="workspace-muted">
                    {f.installed
                      ? f.label
                      : `${f.label} · Server needs framevitals[${f.extra}]`}
                  </p>
                </li>
              ))}
            </ul>
            {!catalog && (
              <p className="workspace-muted mt-3">
                Connect to the API to load the format catalog.
              </p>
            )}
          </details>
        </aside>
      </div>
    </>
  );
}
