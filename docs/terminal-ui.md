# Terminal workspace

FrameVitals includes a zero-extra-dependency local terminal frontend inspired
by the convenience of tools such as Mole: launch one command and navigate
protocols, source files and findings without opening a browser.

## Launch

On macOS/Linux, install the package and use either binary:

```bash
python3 -m pip install -e .
fv
# or:
framevitals tui
```

Running `framevitals` without a subcommand also opens the workspace when
stdin and stdout are connected to a real terminal. To use the simple
line-oriented alternative:

```bash
framevitals ui --plain
```

In CI/non-interactive processes, the command without arguments prints help.
Use `framevitals prism ...` and the other existing subcommands for automation.

## What the terminal can do

| Action | Input | Result |
| --- | --- | --- |
| Prism | Current source, optional reference | Full diagnosis, Beacons, trust/drift when a reference is supplied |
| Tide | Current and reference files | Structural/distribution change |
| Axiom | Reference with optional current source | Infer or validate expectations |
| Pulse | Current source | Compact health snapshot |
| Inspect | Source | Recognition metadata and capabilities |
| Forge preview | Tabular source | Conservative cleaning plan; changes are **not** applied |

The terminal's **L** key displays the complete, live support catalog.
The file explorer marks unsupported extensions and shows which formats need
an additional dependency. You can also list supported formats without a
terminal UI:

```bash
framevitals formats
fv formats --json
```

Supported groups include CSV/TSV/Parquet/Excel, structured JSON/JSONL/YAML/
TOML/Jupyter notebooks, NumPy tensors, GraphML/GEXF/GML, ONNX/Safetensors,
and document formats: PDF, DOCX, PPTX, TXT, Markdown, HTML, XML, and common
source-code text files.

Install document parsers with `pip install "framevitals[documents]"`. PDF
inspection extracts text and metadata from a bounded sample of pages; it does
**not** perform OCR on scanned PDFs. The terminal reads files locally, and
optional parser errors include specific installation guidance.

## Keyboard controls

| Key | Behavior |
| --- | --- |
| Up/Down or J/K | Move through protocols |
| 1–6 | Jump to protocol |
| Enter or R | Run selected protocol |
| P | Browse the local filesystem |
| F | Type or paste a file path |
| B | Set baseline/reference source |
| X | Clear baseline/reference |
| D | Cycle execution depth |
| Tab or Left/Right | Overview, Beacons, Details |
| Page Up/Down, Home/End | Scroll results |
| E | Export JSON or HTML (where supported) |
| L | List all supported file formats and dependency requirements |
| Q | Quit |

When a diagnostic runs, the UI remains responsive and displays a running
indicator. Diagnostic computation runs in a daemon worker rather than writing
to the curses display. The UI does not claim to cancel an in-progress engine
computation; exiting the interactive process terminates the session.

## Safety and privacy

The terminal runs **locally**. It does not send files to a web backend.

- It does not edit or overwrite input data.
- Forge is available as **preview only**.
- Exports require an explicit action.
- `.npy` arrays are loaded with NumPy pickle disabled.
- It does not deserialize untrusted PyTorch pickle checkpoints.
- It does not run model forward/backward passes from a file merely to draw a UI.
- Model libraries like NetworkX/ONNX remain optional; missing dependencies
  are surfaced as errors with guidance.

This interface is a local front end to the existing stable public protocols.
JSON outputs from the direct CLI remain suitable for scripting and CI.
