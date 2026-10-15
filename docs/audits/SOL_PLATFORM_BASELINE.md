# Phase 1 platform baseline — 2026-10-15

Repository: https://github.com/parthdongre/FrameVitals

Working branch: `work/framevitals-sol-platform`.
Baseline: `feature/structured-data-graph-engine` at `009b2e7fb40ec707198cbbf93597b074d442d434`.
`main`: `2a051f1360ba227308279bfbc74fb3eb5e91ebfe` (274 commits behind baseline).
`dev`: `21a43496ab5c3260a894c6fb1d82100492b1b93c` (289 commits behind baseline).
Other branches inspected through refs/tree comparisons: website branches contain deployment/artwork changes; Dependabot branches contain isolated frontend dependency updates. None supersedes the structured-data implementation.

## Verification before implementation

Environment: Linux x86_64, Python 3.12; fresh venv with `pip install -e '.[dev,web,arrow]'`; Node/npm from provided runtime.

| Check | Baseline result |
| --- | --- |
| Clean editable installation / pip check | Passed |
| Python suite | 518 passed, 15 skipped, 1 failed in 21.67 s |
| Existing failure | `test_file_source_reports_cheap_metadata_without_loading`: dependency-sensitive expectation of unknown CSV rows; with Arrow, DelimitedTextSource already counts rows without materialization. The corrected fixture covers installations with and without Arrow |
| Missing optional tests | native extension, DuckDB, Polars, ONNX, PyTorch |
| Frontend production build | Passed; entry JS 447.70 kB / 141.96 kB gzip; lazy Highcharts 286.56 kB / 105.98 kB gzip |
| CLI version, doctor, help | Passed; doctor correctly reports unavailable ONNX |
| CLI/TUI automated coverage | Included in Python suite; actual macOS curses rendering unavailable on Linux |
| API tests | Existing hardening/dependency tests passed; structured uploads unavailable |
| Rust tests | Initially unavailable: cargo/rustc not installed; toolchain installation attempted separately |

Representative fresh-process observations (not comparative scientific benchmarks): import 22.63 ms, max RSS 12,160 KiB; first quick 8×6 tensor Prism 785.35 ms / 215,648 KiB; subsequent 100×2 table 22.64 ms / 220,036 KiB. Tensor dispatch eagerly imports the tabular pipeline before discovering the source.

## Confirmed engineering gaps

- Browser upload allowlist and parser are restricted to CSV/TSV/JSON/Excel; structured engines only exposed in Python/terminal.
- Upload validation errors are collapsed to HTTP 500; malformed/oversized errors can return HTML.
- AI/report/download endpoints lack uniform session ownership. A fresh session can request a known analysis identifier.
- Configured external AI can be invoked without explicit per-request consent.
- Cache eviction drops upload references without deleting retained files; automatic PDF jobs create unbounded threads.
- Result `to_json` can emit nonstandard NaN/Infinity; absent metrics are sometimes presented as zero.
- Graph XML and Office archive parsers need common preflight resource/security checks.
- Flask app lives outside the installed package, so an installed wheel has no supported server command.
- Frontend backend-base handling is duplicated and inconsistent; mobile navigation can overflow.

Mathematical formulas are outside Phase 1 certification. Baseline algorithm behavior is preserved for Astra validation.
