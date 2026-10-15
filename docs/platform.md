# FrameVitals platform guide

FrameVitals runs one Prism protocol over domain-specific engines. Browser and terminal code do not implement a second analytical engine.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[web,graph,documents,excel,arrow,onnx]'
framevitals doctor
framevitals formats
framevitals serve
```

The API binds to `127.0.0.1:5055`. In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/#/analyze`. Vite forwards API calls to port 5055. Nothing is deployed by these commands. Node 22 and Python 3.11+ are supported development choices.

A basic `pip install framevitals` remains a pure Python installation; Rust, Flask, document readers, PyTorch, and plotting packages are not added to its default dependency set. The existing NumPy/Pandas/SciPy/scikit-learn/statsmodels dependencies remain for compatibility; this is not a minimal NumPy-only distribution. Installing `all` includes substantial optional ML dependencies.

## Source journey

Choose a file → server detects the source → inspect capabilities and preview → choose a depth → optionally provide a reference of the same source kind → run → review measurements, findings, methods and limitations → export the complete JSON result.

CSV/TSV previews show six rows and up to forty columns. Tensor previews show shape/dtype without loading pickle objects. Preview is a separate upload; it is removed after the request. The analysis upload is also removed when the unified request completes.

The result page adapts units to the source. Nodes and edges are never presented as table rows and columns. The graph view shows a bounded excerpt of real nodes and edges, using a circular display layout, not an analytical embedding. Expand evidence sections to inspect components, centrality, communities, paths, model weights, document extraction or relational keys. Up to forty children per expansion and six nesting levels are rendered; JSON retains all returned fields. Analytical engines can themselves return bounded summaries.

## Supported inputs

The live catalog from `framevitals formats`, `framevitals capabilities`, or `GET /api/capabilities` is authoritative for parser availability in your installation.

| Input | Prism kind | Optional extra | Browser upload |
| --- | --- | --- | --- |
| CSV, TSV | Tabular | None | Yes |
| Parquet | Tabular | `arrow` | Yes |
| XLSX, XLS | Tabular | `excel` | Yes |
| JSON, JSONL, NDJSON, IPYNB | Nested | None | Yes |
| YAML, YML | Nested | `documents` | Yes |
| TOML | Nested | None (Python 3.11+) | Yes |
| NPY | Tensor | None; objects/pickle disabled | Yes |
| GraphML, GEXF, GML | Graph | `graph` | Yes |
| PDF | Document | `documents` | Yes; text extraction only, no OCR |
| DOCX, PPTX | Document | `documents` | Yes |
| TXT, MD, Markdown, RST, LOG, HTML, HTM, source-code text | Document | None | Yes; scripts/macros never execute |
| XML | Document | `documents` | Yes; DTD/entities disabled |
| ONNX | Model | `onnx` | Yes; no inference or external tensor-file loading |
| Safetensors | Model | None | Yes; bounded metadata/weight diagnostics |
| Pandas/Arrow/Polars tables, NetworkX graphs, NumPy arrays | Domain-specific | Adapter dependencies as applicable | Python objects, not uploaded objects |
| PyTorch modules, runtime batches, gradients, optimizers | Model | Install PyTorch explicitly | In-memory Python only |
| Mapping of table names to DataFrames | Relational | None beyond tabular dependencies | In-memory Python only |
| PT, PTH, PKL, pickle checkpoints, NPZ | Unsupported as uploaded files | — | No |

Prism treats JSON as nested data consistently across Python, terminal and browser. The lower-level legacy `analyze()` API remains tabular and can load tabular JSON explicitly; it is not a generic structured dispatcher.

## Depth and applicability

| Depth | Intended use | Interpretation |
| --- | --- | --- |
| Quick | First inspection | Small budgets, many deliberate skips |
| Standard | Routine diagnostics | Balanced budgets |
| Deep | Larger investigations | Additional work where applicable |
| Research | Largest built-in budgets | Still bounded; not a mathematical certification |

Budgets differ by domain. Availability depends on data, target selection, installed dependencies and budgets. A listed capability is not a promise every diagnostic runs at every depth. `focus` is tabular-only. Structured runtime hooks require an in-memory PyTorch module. A reference must match the source kind; model-file comparisons also require matching model formats.

## Python and terminal

```python
import framevitals as fv
import numpy as np

print(fv.capabilities())
result = fv.prism(np.eye(4), depth='quick')
print(result.summary_text())
envelope = result.to_envelope()
result.to_json('prism.json')
result.to_html('prism.html')
```

```bash
framevitals                       # TUI on a terminal, help when redirected
framevitals tui                   # arrows / Enter; Q quits
framevitals tui --plain           # line-oriented fallback
framevitals formats --json
framevitals capabilities source.npy
framevitals prism source.npy --depth quick --format envelope
framevitals prism notes.pdf --html-report notes.html
framevitals tide reference.json current.json --format json
framevitals axiom reference.csv --format json
framevitals forge data.csv        # preview; --output explicitly applies cleaning
framevitals pulse data.csv --depth quick --format json
```

Use `--help` for a command's exact options. Expected input/dependency failures exit 2 with a readable message. Existing quality-gate exit semantics remain. TUI discovery uses the same source-capability registry; press L for formats. Full-screen curses requires a real terminal, a usable TERM setting and curses support; use `--plain` otherwise. Actual macOS terminal rendering was not tested in the Linux implementation environment.

## Result contract

`PrismResult` and `AnalysisResult` remain mapping-compatible. Existing `to_json()` results keep their schemas. `to_envelope()` adds an independent versioned contract (`schema_version = "1"`) and leaves analytical values unchanged.

An envelope includes source, mode, original analysis, execution metadata, elapsed time when measured, per-diagnostic status, findings, reference change and trust results. HTTP adds generated `id` and display `filename`. Browser filenames and IDs are transport metadata and differ from a direct local Python run.

| Field | Values and meaning |
| --- | --- |
| Diagnostic status | `completed`, `skipped`, `unavailable`, `not_applicable`, `failed`, `partial` |
| Coverage | `full`, `sampled`, `unknown`; separate from computation status |
| Accuracy | `exact`, `approximate`, `unspecified`; exactness requires explicit engine metadata |
| Overall status | `partial` if an explicit failure/truncation is present; completed execution does not mean every optional diagnostic ran |
| Nonfinite values | NaN and infinity serialize as JSON `null`, preserving valid RFC 8259 JSON |
| Missing values | Remain `null`/unavailable; never silently become zero |
| Health score | Heuristic diagnostic signal; not calibrated risk |

Native engine status is retained; no unsupported exactness is inferred from a successful return. Structured aggregate sampling flags still need finer scientific provenance in Phase 2. Inspect individual evidence and methods before interpreting scores.

## HTTP and privacy

- `GET /api/capabilities`: installed format catalog and source capabilities.
- `POST /api/inspect`: multipart `dataset`; returns source and bounded preview.
- `POST /api/prism`: multipart `dataset`, optional `reference`, `analysis_mode`, optional `target_column`; returns the unified envelope.
- Legacy `POST /api/analyze` and its tabular report projection remain for compatibility.
- Cached reports/downloads require the session that created the result. Cookies are HTTP-only and SameSite=Lax.
- AI endpoints require explicit JSON `allow_ai: true`. Configured providers can be external. Legacy AI controls disclose this and require a checkbox. The unified analysis route always disables AI and does not send inputs to providers.
- API/download responses are marked `no-store`. Static chart artifacts are session-checked.

The backend is local by default and is not an account-authenticated hosted service. To deploy privately, put it behind authentication and TLS, configure one stable `FRAMEVITALS_SECRET_KEY` shared by workers, and set `FRAMEVITALS_SECURE_COOKIES=1`. Sessions and caches are process-local: use one worker or sticky sessions for this prototype. Separate-origin deployments need an explicit reverse proxy; the default dev proxy avoids CORS. `VITE_BACKEND_URL` centralizes the optional API origin, and `FRAMEVITALS_ALLOWED_ORIGINS` is a comma-separated server allowlist, not an authentication mechanism.

## Resource limits

- 50 MiB HTTP request limit, including reference and multipart overhead; 96 MiB local file preflight.
- Two concurrent analyses per process; excess requests return JSON HTTP 429.
- Office archives: at most 4,096 entries, 128 MiB total declared uncompressed bytes, 32 MiB per member; traversal paths/encryption rejected. No archives are extracted to disk.
- JSON/JSONL: 32 MiB; YAML/TOML: 16 MiB. Existing document/page/text and algorithm budgets still apply.
- Bounded analysis cache (default 16; configurable up to 128), two PDF workers/four outstanding PDF jobs on the legacy route.

These are application budgets, not a sandbox or OS-enforced CPU/RSS/time limit. Parser libraries can allocate before applying downstream analysis budgets, especially PDF extraction. Use process/container isolation before opening a service to hostile multi-tenant workloads. Legacy generated PDFs/cleaned files/charts need deployment retention management; unified requests create none of those files.

## Verification

```bash
python -m pip install -e '.[dev,web,arrow,onnx]'
python -m pytest -q
python -m ruff check src/framevitals tests app.py --select E9,F401,F821,F822,F823,F841
python -m build
python -m twine check dist/*
cd frontend
npm ci
npm run build
npx playwright install --with-deps chromium
npm test
```

See [the audit](audits/SOL_PLATFORM_AUDIT.md), [baseline](audits/SOL_PLATFORM_BASELINE.md), [module map](audits/SOL_MODULE_MAP.md), and [Astra handoff](work-handoff/SOL_HANDOFF.md).
