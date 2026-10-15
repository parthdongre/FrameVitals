# Sol → Astra: FrameVitals platform handoff

## 1. Repository, branch and immutable pins

- Repository: https://github.com/parthdongre/FrameVitals
- Working branch: `work/framevitals-sol-platform`.
- **Exact final tested implementation commit:** `2cace26cb6a6ed0c79fed165bc6b34a5b6feefd8`.
- Base branch: `feature/structured-data-graph-engine`.
- Exact base SHA: `009b2e7fb40ec707198cbbf93597b074d442d434`.
- Pull request: https://github.com/parthdongre/FrameVitals/pull/66 (unmerged).
- **Final deliverable branch-head SHA:** pinned in the PR description after the final documentation commit. Verify with `git rev-parse origin/work/framevitals-sol-platform` and the PR `head.sha`. A document cannot contain the hash of its own commit; the literal implementation SHA above is immutable, and subsequent handoff commits contain documentation only.

Create Astra's branch from the final PR head, which includes these documents. Do not rebase onto main/dev; those branches are hundreds of commits behind the verified implementation baseline. GitHub commits were published through the connected API because shell Git had no push credentials. The published trees were checked against the local tested trees; no remote history was rewritten.

## 2. Architecture

`Input → core/source + file_formats → capabilities → Prism → domain engine/planner → AnalysisResult → additive result envelope → report/UI`.

Prism owns domain dispatch. Tabular execution continues through `analysis_api` / `sources` / `pipeline` or streaming execution. Structured sources use `structured_analysis` and individual `analysis/*` modules. Browser routes invoke the same Prism call. Python, CLI, TUI and browser share format/capability discovery; UI code contains presentation logic only.

The root `app.py` is a compatibility alias for `framevitals.webapp`, now included in installed wheels. The old tabular `/api/analyze` projection remains under compatibility tests. Domain-specific data is never forced into a DataFrame for the new browser workflow.

## 3. Public API changes

Additive APIs:

- `framevitals.capabilities(data=None)` — live catalog or source applicability, without implicit package installation.
- `AnalysisResult.to_envelope()` / `PrismResult.to_envelope()` — v1 transport contract.
- `framevitals capabilities [file]` — JSON capability discovery.
- `framevitals prism FILE --format envelope`.
- `framevitals serve [--port 5055]` — local API, requires `web` extra.
- HTTP `GET /api/capabilities`, `POST /api/inspect`, `POST /api/prism`.

Existing dictionary results and `to_json()` schemas remain. Nonfinite JSON values now become `null`; unsafe/nonstandard NaN/Infinity output is intentionally corrected. Expected CLI input/dependency errors produce readable stderr and exit 2. Structured Prism rejects an invalid depth instead of silently selecting Standard.

AI endpoints require `allow_ai: true`. Result downloads/status/static charts now require the owning session. These are intentional security boundary changes. Origin checks and concurrency limits can return 403/429 respectively.

## 4. Result schema and compatibility

`result_contract.py` defines additive envelope `schema_version="1"`; original `result_schema_version="1"` mapping payloads remain intact. The envelope preserves source, original analysis, original execution metadata, findings, trust and reference change. HTTP adds transport ID/display filename and measured elapsed time.

Computation statuses: completed, skipped, unavailable, not_applicable, failed, partial. Coverage: full/sampled/unknown. Accuracy: exact/approximate/unspecified. Status, sampling and numerical accuracy are separate concepts. Successful computation does not imply scientific exactness. Missing values remain unavailable/null, never zero. Scores are explicitly heuristic.

New graph `topology_preview` contains at most 48 actual nodes, 120 actual edges, a 5000-edge scan cap, a partial flag and a display-only circular layout label. Tensor matrix payloads include up to 64 already-computed singular values for charts. No spectral calculation or graph algorithm changed to produce those presentation fields.

## 5. Source adapters and formats

See the authoritative [format matrix](../platform.md#supported-inputs). All catalogued file formats are exposed to unified HTTP upload with optional parser checks; generated storage names preserve safe suffixes and user filenames remain display metadata.

Actual HTTP tests cover CSV, TSV, JSON, JSONL, YAML, TOML, Markdown, HTML, NPY, GraphML, PDF, DOCX, PPTX, XLSX, Parquet, Safetensors and ONNX. Existing tests cover additional GraphML/GEXF/GML, XML and document behaviors. Excel XLS uses the optional xlrd reader. JSON/IPYNB are nested sources for Prism. In-memory PyTorch modules/runtime batches and relational DataFrame mappings remain Python-only. Pickle-based files and NPZ are not advertised as supported uploads.

## 6. Test commands and observed results

Environment: Linux x86_64, Python 3.12, Node runtime, Rust stable. Fresh editable environment plus verification-only ONNX, DuckDB, Polars, CPU PyTorch and a built native extension.

| Check | Result |
| --- | --- |
| Baseline Python | 518 passed, 15 skipped, 1 stale CSV-metadata assertion failed |
| Final full Python suite | 574 passed; no skips in the expanded verification environment |
| Shared platform regression file | 31 passed |
| Browser | 4 Chromium end-to-end tests passed; real Flask/Vite requests, export, reference comparison, graph/tensor views, mobile/error cases |
| Accessibility | axe scans of tested main-content workspace/results passed; not whole-product WCAG certification |
| Frontend TypeScript/production build | Passed |
| Python critical/unused lint, compile, diff whitespace | Passed |
| Rust `cargo test --workspace` | Passed: 20 core tests, bridge/doc targets successful |
| Rust standalone default core tests | 15 passed |
| Rust fmt / Clippy all targets with warnings denied | Passed |
| Maturin PyO3 native build + Python native tests | Passed |
| Pure wheel + sdist / Twine metadata | Passed; pure wheel contains no native binary and includes packaged web API |
| Separate clean core-only wheel environment | Import, tensor Prism, CLI and native-free fallback passed; test suite 462 passed / 63 optional skips |
| Strict MkDocs build | Passed |
| Actual Linux TUI PTY entry/quit | Exit 0; alternate-screen sequence observed |
| macOS/Windows live terminals, Safari | Not executed locally; cross-platform CI retained |

```bash
python -m pip install -e '.[dev,web,arrow,onnx]'
# Optional verification integrations: install duckdb, polars, CPU torch explicitly.
python -m pytest -q
python -m ruff check src/framevitals tests app.py --select E9,F401,F821,F822,F823,F841
python -m compileall -q src/framevitals app.py
python -m build
python -m twine check dist/*
cd frontend
npm ci
npm run build
npx playwright install --with-deps chromium
npm test
```

Known numerical warnings remain: NetworkX assortativity on degenerate graphs and sklearn's deliberately bounded neural-fit convergence warning. They are visible, not suppressed to conceal failures. New CI `platform-workflows.yml` runs API/TUI/browser checks and uploads screenshots. The first CI run exposed two test assumptions about optional dependencies: Arrow-only CSV row counts and an XML parser assumed present in the core-only environment. Both tests now explicitly verify parser absence/fallback rather than skipping those assertions. Remote CI status is tracked in the PR description independently of local results.

## 7. Rust build instructions

The default Python wheel uses setuptools and requires no Rust. For native development:

```bash
python -m pip install maturin
# Activate the intended Python virtual environment first.
maturin develop --manifest-path rust/framevitals-py/Cargo.toml
cd rust
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

PyO3 targets abi3 Python 3.11+ and Arrow interoperability. The implementation environment had a stale Python shared-library symlink and `/install/lib` in sysconfig. Local verification used a scratch symlink to the actual `libpython3.12.so.1.0` plus `RUSTFLAGS='-L native=<scratch-library-directory>'` / `LD_LIBRARY_PATH` for linking. This was an environment repair; no Rust source/linker workaround was committed. A normal Python development installation should provide its shared library directly.

## 8. Frontend build and runtime

React 19, TypeScript, Vite 6, Tailwind, existing Framer Motion/React Query and lazy legacy Highcharts. New source views use lightweight SVG/CSS with real engine outputs. Page chunks are lazy. The API base is centralized in `lib/api.ts`; by default relative requests use the Vite same-origin proxy. Production separate-origin access needs an explicit credential-aware reverse proxy.

State is in memory; refreshing loses the latest result. JSON export preserves the returned payload. The legacy detailed tabular report components remain available to compatibility callers. Legacy AI controls require explicit provider consent; unified Prism never calls AI.

## 9. Benchmark commands and observations

```bash
python benchmarks/platform_probe.py --kind import
python benchmarks/platform_probe.py --kind tensor
python benchmarks/platform_probe.py --kind tabular
python benchmarks/platform_probe.py --kind graph
```

Run each command in a fresh process. The probe's tracemalloc peak covers Python allocations only. Baseline import: 22.63 ms / 12160 KiB RSS. Baseline first quick 8×6 tensor: 785.35 ms / 215648 KiB RSS. After lazy structured dispatch, a representative same-shape run observed 12.12 ms / 32912 KiB RSS without importing pipeline or sklearn. These are environment observations, not rigorous algorithm-speed claims.

Frontend entry JS: 447.70 → 382.77 kB (141.96 → 122.12 kB gzip). Highcharts remains a separate lazy 286.56 kB chunk. No new runtime visualization dependency was added. Pure wheel approximately 382 kB compressed; scientific dependencies remain larger and unchanged by default.

## 10. Remaining engineering limitations

- Local server is not account-authenticated or a production multi-tenant service. Deploy only behind authentication/TLS with a stable secret; do not assume origin checks are authentication.
- Cache/session artifacts are process-local. Use a single worker or sticky sessions until a shared state design is implemented.
- Parsing limits are application-level; no OS CPU/RSS deadline or parser sandbox. PDF extraction may allocate/decompress before text truncation. Adversarial deployments require isolated workers.
- Unified requests delete raw uploads, but abrupt process termination can leave temporary files. Legacy generated charts/PDFs/cleaned datasets need retention cleanup.
- Browser does not create arbitrary Python models, relational table mappings, runtime hooks or optimizers from uploaded executable checkpoints.
- Accessibility/mobile checks are scoped to tested Chromium screens. All combinations of legacy tabs and browser engines were not manually certified.
- No release, merge or deployment was performed.

## 11. Mathematical work for Astra

Use [ASTRA_SCIENCE_BACKLOG.md](ASTRA_SCIENCE_BACKLOG.md). Highest priorities: strided spectral aliasing/denominators, nonfinite sample coverage, context-dependent score calibration, graph/path metric provenance, drift validity, ML leakage/evaluation and relational inference precision.

## 12. Mathematical modules intentionally preserved

All scientific formulas, penalties, statistical thresholds, estimators, path algorithms, training algorithms and Rust kernels were left unchanged. Changes inside `analysis/graph.py` are parser security preflight and bounded visualization data; inside `analysis/tensor.py` only existing singular values are exposed; inside `analysis/document.py` only input safety preflight was added. Domain engines, predictive diagnostics, health scoring, anomaly ensembles, drift methods, sketches and runtime-model logic otherwise retain baseline calculations.

## 13. Scientific fixtures and reproduction harnesses

- `tests/fixtures/science/cases.json`: eight source/generator-based validation cases with explicit questions.
- `tests/test_diagnostic_safety.py`: existing model/graph safety regressions.
- `tests/test_structured_prism.py`, `test_structured_extended.py`: source/domain integration fixtures.
- `tests/test_platform_workflows.py`: cross-interface result agreement, strict serialization, missing-score HTML, parser guards, session isolation, cleanup, optional binary formats.
- `frontend/tests/workspace.spec.ts`: real HTTP workflow and bounded visualization regressions.
- `benchmarks/platform_probe.py`: fresh-process platform overhead; existing benchmark scripts remain available for scientific benchmarking.

Golden scientific oracles should be added by Astra; Phase 1 integration tests do not constitute full mathematical validation.

## 14. Resource limits and performance

HTTP 50 MiB per multipart request; local preflight 96 MiB; Office 4096 entries / 128 MiB decompressed total / 32 MiB member; structured parser and per-domain analysis limits retained. Two analyses per process; bounded cache; two legacy PDF workers/four outstanding jobs. Browser evidence trees render at most forty children per branch and six levels. See the platform guide for exact qualifications and privacy behavior.

## 15. Building on this work

1. Fetch PR #66 and verify the final head SHA in its description against the branch.
2. Create `work/framevitals-astra-science` from that exact head, with the handoff docs present.
3. Reproduce the environment and baseline tests before modifying numerical code.
4. Address backlog issues with analytic fixtures and explicit population/method definitions.
5. Keep versioned metadata additive and preserve missing-versus-zero distinctions.
6. Run engineering, native, browser and benchmark checks after numerical/schema changes.
7. Update score methodology/provenance documentation whenever interpretation changes. Do not relabel sampled/approximate metrics exact without evidence.
